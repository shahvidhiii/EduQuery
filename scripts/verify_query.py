
import chromadb
import os
import requests
from dotenv import load_dotenv

load_dotenv()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", os.path.join(base_dir, "data", "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "video_chunks")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

def get_embedding(text):
    r = requests.post(f"{OLLAMA_HOST}/api/embeddings", json={"model": EMBEDDING_MODEL, "prompt": text})
    return r.json()["embedding"]

def test_query(query_text):
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    emb = get_embedding(query_text)
    results = collection.query(query_embeddings=[emb], n_results=5, include=["documents", "metadatas"])
    
    print(f"Results for: '{query_text}'\n")
    for i in range(len(results["ids"][0])):
        print(f"Rank {i+1}:")
        print(f"  ID: {results['ids'][0][i]}")
        print(f"  Doc: {results['documents'][0][i][:100]}...")
        print(f"  Meta: {results['metadatas'][0][i]}")
        print("-" * 20)

if __name__ == "__main__":
    test_query("environment day?")
