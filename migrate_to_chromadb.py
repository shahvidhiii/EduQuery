"""
migrate_to_chromadb.py — Migrate embeddings.pkl → ChromaDB

This script reads the existing pickle-based embeddings and upserts them
into a ChromaDB persistent collection. This is a one-time migration step
to upgrade from the legacy pickle backend to ChromaDB.

Usage:
    python migrate_to_chromadb.py            # Migrate (skip existing)
    python migrate_to_chromadb.py --reset    # Wipe collection and re-migrate
    python migrate_to_chromadb.py --dry-run  # Preview without writing

After migration, set USE_CHROMADB=true in .env to switch the backend.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import chromadb
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(base_dir, "embeddings.pkl")
_chroma_dir = os.getenv("CHROMA_DB_DIR", "chroma_db")
CHROMA_DB_DIR = _chroma_dir if os.path.isabs(_chroma_dir) else os.path.join(base_dir, _chroma_dir)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "video_chunks")


def migrate(reset: bool = False, dry_run: bool = False):
    print("=" * 55)
    print("   EduQuery — Pickle → ChromaDB Migration")
    print("=" * 55)
    print(f"Source      : {EMBEDDINGS_FILE}")
    print(f"Destination : {CHROMA_DB_DIR}")
    print(f"Collection  : {COLLECTION_NAME}")
    print()

    # Check source exists
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"✗ ERROR: {EMBEDDINGS_FILE} not found.")
        print("  Run read_chunks.py first to create embeddings.")
        sys.exit(1)

    # Load pickle
    print("Loading embeddings.pkl...")
    start = time.time()
    df = pd.read_pickle(EMBEDDINGS_FILE)
    df["embedding"] = df["embedding"].apply(
        lambda x: list(x)
        if (x is not None and hasattr(x, "__iter__") and not isinstance(x, (str, bytes)))
        else []
    )
    print(f"✓ Loaded {len(df)} chunks in {time.time() - start:.1f}s")
    print()

    # Preview columns
    print(f"Columns: {list(df.columns)}")
    print(f"Sample row:")
    if len(df) > 0:
        sample = df.iloc[0]
        print(f"  number: {sample.get('number', '?')}")
        print(f"  title : {sample.get('title', '?')}")
        print(f"  start : {sample.get('start', 0)}")
        print(f"  end   : {sample.get('end', 0)}")
        print(f"  text  : {str(sample.get('text', ''))[:80]}...")
        emb = sample.get("embedding", [])
        print(f"  emb   : {len(emb) if isinstance(emb, list) else '?'} dimensions")
    print()

    if dry_run:
        print("── DRY RUN: No changes will be written ──")
        print(f"Would migrate {len(df)} chunks to ChromaDB.")
        return

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("⟳ Existing collection deleted (--reset)")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing = collection.count()
    print(f"  Existing documents: {existing}")
    print()

    # Migrate in batches
    print("Migrating...")
    added = 0
    skipped = 0
    errors = 0
    batch_size = 100

    ids_batch, emb_batch, doc_batch, meta_batch = [], [], [], []

    for idx, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        emb = row.get("embedding")
        if emb is None or (isinstance(emb, list) and len(emb) == 0):
            errors += 1
            continue

        chunk_id = f"v{row['number']}_{float(row['start']):.2f}"

        # Check for existing (only if not reset)
        if not reset:
            try:
                ex = collection.get(ids=[chunk_id])
                if ex and ex["ids"]:
                    skipped += 1
                    continue
            except Exception:
                pass

        emb_list = list(emb) if not isinstance(emb, list) else emb

        ids_batch.append(chunk_id)
        emb_batch.append(emb_list)
        doc_batch.append(text)
        meta_batch.append({
            "video_number": str(row["number"]),
            "title": str(row["title"]),
            "start_time": float(row["start"]),
            "end_time": float(row.get("end", 0)),
        })

        if len(ids_batch) >= batch_size:
            collection.upsert(
                ids=ids_batch,
                embeddings=emb_batch,
                documents=doc_batch,
                metadatas=meta_batch,
            )
            added += len(ids_batch)
            print(f"  ↑ Migrated {added} chunks...")
            ids_batch, emb_batch, doc_batch, meta_batch = [], [], [], []

    # Flush remaining
    if ids_batch:
        collection.upsert(
            ids=ids_batch,
            embeddings=emb_batch,
            documents=doc_batch,
            metadatas=meta_batch,
        )
        added += len(ids_batch)

    print()
    print("=" * 55)
    print("  Migration Complete!")
    print(f"  Added   : {added}")
    print(f"  Skipped : {skipped} (already in DB)")
    print(f"  Errors  : {errors} (empty embeddings)")
    print(f"  Total   : {collection.count()} documents")
    print("=" * 55)
    print()
    print("Next steps:")
    print("  1. Set USE_CHROMADB=true in .env")
    print("  2. Restart the server: python main.py")
    print()


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    dry_run_flag = "--dry-run" in sys.argv
    migrate(reset=reset_flag, dry_run=dry_run_flag)
