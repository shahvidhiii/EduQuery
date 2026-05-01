"""
read_chunks_v2.py — Ingest video transcript chunks into ChromaDB.

This replaces the old pickle-based pipeline (read_chunks.py → embeddings.pkl).
It reads JSON files from the jsons/ folder and stores them in a ChromaDB
collection with Ollama-generated embeddings.

Usage:
    python read_chunks_v2.py          # Ingest all JSON files
    python read_chunks_v2.py --reset  # Wipe the collection and re-ingest
"""

import os
import sys
import json
import time
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
JSONS_FOLDER = os.getenv("JSON_DIR", os.path.join(data_dir, "jsons"))
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", os.path.join(data_dir, "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "video_chunks")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")


def create_embedding(text: str) -> list | None:
    """Call Ollama to create an embedding for a single string."""
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60,
        )
        r.raise_for_status()
        embedding = r.json().get("embedding")
        if not embedding:
            print(f"  [!] No embedding returned for: {text[:60]}...")
            return None
        return embedding
    except requests.exceptions.ConnectionError:
        print("\n--- CRITICAL: Cannot connect to Ollama. Is it running? ---")
        sys.exit(1)
    except Exception as e:
        print(f"  [!] Embedding error: {e}")
        return None


def ingest(reset: bool = False):
    """Read all JSON chunk files and upsert into ChromaDB."""
    print(f"ChromaDB directory : {CHROMA_DB_DIR}")
    print(f"Collection name    : {COLLECTION_NAME}")
    print(f"Embedding model    : {EMBEDDING_MODEL}")

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[RESET] Existing collection deleted (--reset).")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )

    existing_count = collection.count()
    print(f"Existing documents in collection: {existing_count}")

    # --- Load JSON files ---
    if not os.path.isdir(JSONS_FOLDER):
        print(f"ERROR: '{JSONS_FOLDER}' not found. Run create_chunks.py first.")
        sys.exit(1)

    json_files = sorted(f for f in os.listdir(JSONS_FOLDER) if f.endswith(".json"))
    if not json_files:
        print("No JSON files found. Nothing to ingest.")
        return

    total_added = 0
    total_skipped = 0

    for json_file in json_files:
        filepath = os.path.join(JSONS_FOLDER, json_file)
        with open(filepath, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"\n--- {json_file} ({len(chunks)} chunks) ---")

        ids_batch = []
        embeddings_batch = []
        documents_batch = []
        metadatas_batch = []

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            # Deterministic ID: <video_number>_<start_time>
            chunk_id = f"v{chunk['number']}_{chunk['start']:.2f}"

            # Skip if already in DB
            existing = collection.get(ids=[chunk_id])
            if existing and existing["ids"]:
                total_skipped += 1
                continue

            embedding = create_embedding(text)
            if embedding is None:
                continue

            ids_batch.append(chunk_id)
            embeddings_batch.append(embedding)
            documents_batch.append(text)
            metadatas_batch.append({
                "video_number": str(chunk["number"]),
                "title": chunk["title"],
                "start_time": float(chunk["start"]),
                "end_time": float(chunk["end"]),
                "source_file": json_file,
            })

            # Batch upsert every 50 chunks
            if len(ids_batch) >= 50:
                collection.upsert(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    documents=documents_batch,
                    metadatas=metadatas_batch,
                )
                total_added += len(ids_batch)
                print(f"  ^ Upserted {total_added} chunks so far...")
                ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []

        # Flush remaining
        if ids_batch:
            collection.upsert(
                ids=ids_batch,
                embeddings=embeddings_batch,
                documents=documents_batch,
                metadatas=metadatas_batch,
            )
            total_added += len(ids_batch)

        print(f"  [OK] {json_file} done")

    print(f"\n{'='*50}")
    print(f"Ingestion complete!")
    print(f"  Added   : {total_added}")
    print(f"  Skipped : {total_skipped} (already in DB)")
    print(f"  Total   : {collection.count()} documents in collection")
    print(f"{'='*50}")


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    ingest(reset=reset_flag)
