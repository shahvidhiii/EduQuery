import pandas as pd
import os
import sys
import chromadb
import requests
import json

# Configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(base_dir, "embeddings.pkl")
CHROMA_DB_DIR = os.path.join(base_dir, "chroma_db")
COLLECTION_NAME = "video_chunks"
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"

def migrate_pickle_to_chromadb():
    """Migrate embeddings.pkl to ChromaDB collection."""
    
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"❌ Error: {EMBEDDINGS_FILE} not found")
        return 0, 0
    
    print(f"Loading embeddings.pkl...")
    try:
        df = pd.read_pickle(EMBEDDINGS_FILE)
        print(f"✅ Loaded {len(df)} chunks from embeddings.pkl")
    except Exception as e:
        print(f"❌ Error loading embeddings.pkl: {e}")
        return 0, 0
    
    # Check what's in the embeddings file
    print(f"\nEmbeddings content:")
    print(f"  Unique video numbers: {sorted(df['number'].unique())}")
    print(f"  Total chunks: {len(df)}")
    print(f"  Chunk distribution: {df['number'].value_counts().sort_index().to_dict()}")
    
    # Connect to ChromaDB
    print(f"\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    existing_count = collection.count()
    print(f"Existing documents in ChromaDB: {existing_count}")
    
    # Prepare for migration
    added = 0
    skipped = 0
    ids_batch = []
    embeddings_batch = []
    documents_batch = []
    metadatas_batch = []
    
    print(f"\nMigrating embeddings...")
    for idx, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            skipped += 1
            continue
        
        emb = row.get("embedding")
        if emb is None or (hasattr(emb, "__len__") and len(emb) == 0):
            skipped += 1
            continue
        
        # Create unique ID for chunk
        chunk_id = f"v{row['number']}_{float(row['start']):.2f}"
        
        # Check if already exists
        try:
            existing = collection.get(ids=[chunk_id])
            if existing and existing["ids"]:
                skipped += 1
                continue
        except:
            pass
        
        # Prepare embedding (ensure it's a list)
        emb_list = list(emb) if not isinstance(emb, list) else emb
        
        ids_batch.append(chunk_id)
        embeddings_batch.append(emb_list)
        documents_batch.append(text)
        metadatas_batch.append({
            "video_number": str(row["number"]),
            "title": str(row["title"]),
            "start_time": float(row["start"]),
            "end_time": float(row.get("end", 0)),
        })
        
        # Batch insert every 50 documents
        if len(ids_batch) >= 50:
            try:
                collection.upsert(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    documents=documents_batch,
                    metadatas=metadatas_batch,
                )
                added += len(ids_batch)
                print(f"  ✓ Migrated {added} documents so far...")
                ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []
            except Exception as e:
                print(f"  ⚠ Error during batch insert: {e}")
    
    # Flush remaining
    if ids_batch:
        try:
            collection.upsert(
                ids=ids_batch,
                embeddings=embeddings_batch,
                documents=documents_batch,
                metadatas=metadatas_batch,
            )
            added += len(ids_batch)
        except Exception as e:
            print(f"  ⚠ Error during final batch insert: {e}")
    
    return added, skipped

if __name__ == "__main__":
    print("=" * 60)
    print("  EMBEDDINGS.PKL → ChromaDB Migration")
    print("=" * 60)
    print()
    
    added, skipped = migrate_pickle_to_chromadb()
    
    print()
    print("=" * 60)
    print("Migration Complete!")
    print(f"  Added:  {added}")
    print(f"  Skipped: {skipped}")
    print()
    
    # Verify
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    final_count = collection.count()
    print(f"Total documents in ChromaDB: {final_count}")
    print("=" * 60)
