import requests
import json
import chromadb

# Test the semantic search directly
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "video_chunks"

print("=" * 70)
print("  Debugging: Why is 'error correction' returning wrong video?")
print("=" * 70)
print()

# Get query embedding
print("Step 1: Create embedding for query 'What is error correction?'")
r = requests.post(
    f"{OLLAMA_HOST}/api/embeddings",
    json={"model": EMBEDDING_MODEL, "prompt": "What is error correction?"},
    timeout=60
)
query_embedding = r.json().get("embedding")
print(f"✓ Query embedding created ({len(query_embedding)} dimensions)")
print()

# Search ChromaDB
print("Step 2: Search ChromaDB (top 10 results)")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    include=["documents", "metadatas", "distances"]
)

print(f"Found {len(results['ids'][0])} results:")
print()
for i, doc_id in enumerate(results['ids'][0]):
    meta = results['metadatas'][0][i]
    doc = results['documents'][0][i]
    distance = results['distances'][0][i]  # Lower distance = better match
    video = meta['video_number']
    title = meta['title'][:40]
    text_preview = doc[:80]
    
    print(f"{i+1}. Video {video} | Distance: {distance:.4f}")
    print(f"   Title: {title}...")
    print(f"   Text: {text_preview}...")
    print()

print("=" * 70)
print("Analysis:")
print("  - Results ranked by semantic similarity (lower distance = better)")
print("  - Expected: Videos 27-30 (Hamming, error detection codes)")
print("  - Actual: Checking above...")
print("=" * 70)
