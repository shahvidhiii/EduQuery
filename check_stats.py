
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "video_chunks")

def stats():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Total chunks in DB: {count}")
        
        # Get all metadatas to check unique numbers
        res = collection.get(include=["metadatas"])
        nums = set()
        if res and res["metadatas"]:
            for m in res["metadatas"]:
                nums.add(m.get("video_number"))
        
        print(f"\nUnique video numbers in DB: {sorted(list(nums))}")
        
        for n in sorted(list(nums)):
            sub_res = collection.get(where={"video_number": n})
            print(f"  Video {n}: {len(sub_res['ids'])} chunks")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    stats()
