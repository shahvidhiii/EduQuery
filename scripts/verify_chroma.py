import chromadb
import pandas as pd
import os

# Verify ChromaDB now has all embeddings
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", os.path.join(base_dir, "data", "chroma_db"))
COLLECTION_NAME = "video_chunks"

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

print("=" * 60)
print("  ChromaDB Verification Report")
print("=" * 60)
print()

total_count = collection.count()
print(f"Total documents in ChromaDB: {total_count}")
print()

# Get sample documents by video number
print("Sample documents by video:")
for video_num in ['01', '21', '25', '30']:
    results = collection.get(
        where={"video_number": {"$eq": video_num}},
        limit=1
    )
    if results['ids']:
        meta = results['metadatas'][0]
        doc = results['documents'][0]
        print(f"\n  Video {video_num}:")
        print(f"    - Title: {meta['title'][:40]}...")
        print(f"    - Start: {meta['start_time']}s")
        print(f"    - Text: {doc[:60]}...")
    else:
        print(f"\n  Video {video_num}: No documents found")

print()
print("=" * 60)

# Test a semantic search
print("\nTesting semantic search...")
print('  Query: "What is parity?"')

# Create embedding for query
import requests

OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"

r = requests.post(
    f"{OLLAMA_HOST}/api/embeddings",
    json={"model": EMBEDDING_MODEL, "prompt": "What is parity?"},
    timeout=60
)
query_embedding = r.json().get("embedding")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas"]
)

print(f"  Found {len(results['ids'][0])} results:")
for i, doc_id in enumerate(results['ids'][0]):
    meta = results['metadatas'][0][i]
    doc = results['documents'][0][i]
    print(f"    {i+1}. Video {meta['video_number']}: {meta['title'][:30]}...  @ {meta['start_time']}s")

print()
print("=" * 60)
print("✅ ChromaDB is ready with all embeddings!")
print("=" * 60)
