import os
import uvicorn
import pandas as pd
import numpy as np
import json
import re
import requests
from urllib.parse import quote
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(base_dir, "embeddings.pkl")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- ChromaDB Configuration ---
USE_CHROMADB = os.getenv("USE_CHROMADB", "false").lower() == "true"
_chroma_dir = os.getenv("CHROMA_DB_DIR", "chroma_db")
CHROMA_DB_DIR = _chroma_dir if os.path.isabs(_chroma_dir) else os.path.join(base_dir, _chroma_dir)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "video_chunks")
_video_dir = os.getenv("VIDEO_DIR", "videos")
VIDEO_DIR = _video_dir if os.path.isabs(_video_dir) else os.path.join(base_dir, _video_dir)

# --- Pydantic Models ---

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = None


class Source(BaseModel):
    video_number: str
    title: str
    start_time: float
    end_time: Optional[float] = None
    video_url: Optional[str] = None


# --- Helper Functions ---

def create_embedding(text: str):
    """Call local Ollama to create an embedding for a single string."""
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60,
        )
        r.raise_for_status()
        embedding = r.json().get("embedding")
        return embedding, None
    except requests.exceptions.ConnectionError:
        print("\n--- ERROR: Could not connect to Ollama. Is it running? ---")
        return None, "OLLAMA_NOT_RUNNING"
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None, str(e)


def ask_llm_stream(context: str, question: str, history: Optional[List[dict]] = None):
    """
    Stream tokens from local Ollama LLM.
    Yields individual text chunks as they arrive.
    """
    history_block = ""
    if history and isinstance(history, list):
        last_msgs = history[-10:]
        history_lines = []
        for m in last_msgs:
            role = m.get("role", "user") if isinstance(m, dict) else getattr(m, "role", "user")
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            history_lines.append(f"[{role.upper()}] {content}")
        history_block = "\n".join(history_lines)

    prompt = f"""
    You are a helpful AI teaching assistant for a video lecture course.
    Use the conversation history and context below to answer the student's question.
    Reference specific video numbers and timestamps when possible.
    
    CONVERSATION HISTORY:
    {history_block}

    CONTEXT FROM VIDEO LECTURES:
    {context}

    STUDENT'S QUESTION:
    {question}

    Provide a clear, helpful answer. When referencing content, mention the video number and approximate timestamp.
    
    ANSWER:
    """

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
    except Exception as e:
        yield f"\n\n⚠️ Error communicating with LLM: {e}"


def find_video_file(video_number: str, title: str) -> Optional[str]:
    """
    Try to find the actual video file for a given video number/title.
    Looks in VIDEO_DIR for files matching common naming conventions.
    """
    if not os.path.isdir(VIDEO_DIR):
        return None

    try:
        files = os.listdir(VIDEO_DIR)
    except Exception:
        return None

    padded = video_number.zfill(2)
    title_lower = title.lower().replace(" ", "").replace("-", "").replace("_", "") if title else ""

    for f in files:
        if not f.lower().endswith((".mp4", ".webm", ".mkv", ".avi", ".mov")):
            continue
        name_lower = f.lower()
        # Match by video number in filename
        if video_number in f or padded in f:
            return f
        # Match by title similarity
        if title_lower and title_lower[:15] in name_lower.replace(" ", "").replace("-", "").replace("_", ""):
            return f

    return None


def build_video_url(video_number: str, title: str, start_time: float) -> Optional[str]:
    """Build a URL for the video file with timestamp parameter."""
    filename = find_video_file(video_number, title)
    if filename:
        return f"/videos/{quote(filename)}#t={int(start_time)}"
    return None


# --- ChromaDB search function ---
def search_chromadb(question: str, top_k: int = 3):
    """
    Query ChromaDB collection for the most relevant chunks.
    Returns (results_list, error_string).
    """
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION)
    except Exception:
        return None, f"ChromaDB collection '{CHROMA_COLLECTION}' not found. Run read_chunks_v2.py or use /api/migrate first."

    question_embedding, err = create_embedding(question)
    if err:
        return None, err

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            text = results["documents"][0][i]
            output.append({
                "video_number": meta.get("video_number", "??"),
                "title": meta.get("title", "Untitled"),
                "start_time": meta.get("start_time", 0.0),
                "end_time": meta.get("end_time", 0.0),
                "text": text,
            })

    return output, None


# --- Pickle-based search function (legacy) ---
def search_pickle(question: str, df, all_embeddings, top_k: int = 3):
    """Search using the legacy pickle-based embeddings."""
    question_embedding, err = create_embedding(question)
    if err:
        return None, err

    q_emb = np.array(question_embedding, dtype=np.float32)
    q_emb /= np.linalg.norm(q_emb) or 1.0
    similarities = np.dot(all_embeddings, q_emb).flatten()
    max_indx = similarities.argsort()[::-1][:top_k]
    results_df = df.loc[max_indx]

    output = []
    for _, row in results_df.iterrows():
        output.append({
            "video_number": str(row["number"]),
            "title": row["title"],
            "start_time": float(row["start"]),
            "end_time": float(row.get("end", 0)),
            "text": row["text"],
        })

    return output, None


# --- Migrate pickle → ChromaDB ---
def migrate_pickle_to_chromadb():
    """
    Read embeddings.pkl and upsert all rows into ChromaDB.
    Returns (added_count, skipped_count, error_string).
    """
    import chromadb

    if not os.path.exists(EMBEDDINGS_FILE):
        return 0, 0, "embeddings.pkl not found"

    try:
        df = pd.read_pickle(EMBEDDINGS_FILE)
    except Exception as e:
        return 0, 0, f"Could not read embeddings.pkl: {e}"

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    added = 0
    skipped = 0

    ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []

    for idx, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        emb = row.get("embedding")
        if emb is None or (hasattr(emb, "__len__") and len(emb) == 0):
            continue

        chunk_id = f"v{row['number']}_{float(row['start']):.2f}"

        # Check if already exists
        existing = collection.get(ids=[chunk_id])
        if existing and existing["ids"]:
            skipped += 1
            continue

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

        if len(ids_batch) >= 50:
            collection.upsert(
                ids=ids_batch,
                embeddings=embeddings_batch,
                documents=documents_batch,
                metadatas=metadatas_batch,
            )
            added += len(ids_batch)
            ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []

    # Flush remaining
    if ids_batch:
        collection.upsert(
            ids=ids_batch,
            embeddings=embeddings_batch,
            documents=documents_batch,
            metadatas=metadatas_batch,
        )
        added += len(ids_batch)

    return added, skipped, None


# --- FastAPI App Setup ---

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 55)
    print("   EduQuery — AI Teaching Assistant API")
    print("=" * 55)

    # Determine backend
    if USE_CHROMADB:
        print(f"→ ChromaDB backend requested (dir: {CHROMA_DB_DIR})")
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            collection = client.get_collection(name=CHROMA_COLLECTION)
            count = collection.count()
            print(f"  ✓ Collection '{CHROMA_COLLECTION}' loaded: {count} documents")
            app_state["backend"] = "chromadb"
        except Exception as e:
            print(f"  ⚠ ChromaDB not ready: {e}")
            print("  → Falling back to pickle backend...")
            app_state["backend"] = "pickle"
    else:
        app_state["backend"] = "pickle"

    # Load pickle as fallback or primary
    if app_state["backend"] == "pickle":
        print("→ Using pickle-based embeddings backend")
        if not os.path.exists(EMBEDDINGS_FILE):
            print(f"  ✗ Embeddings file not found: {EMBEDDINGS_FILE}")
            print("  Run read_chunks.py first to create embeddings.")
        else:
            try:
                df = pd.read_pickle(EMBEDDINGS_FILE)
                df["embedding"] = df["embedding"].apply(
                    lambda x: list(x)
                    if (x is not None and hasattr(x, "__iter__") and not isinstance(x, (str, bytes)))
                    else []
                )
                all_embeddings = np.vstack(df["embedding"].values).astype(np.float32)
                norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                all_embeddings = all_embeddings / norms
                app_state["df"] = df
                app_state["all_embeddings"] = all_embeddings
                print(f"  ✓ Embeddings loaded: {len(df)} chunks")
            except Exception as e:
                print(f"  ✗ Could not load embeddings: {e}")

    # Check video directory
    if os.path.isdir(VIDEO_DIR):
        video_count = len([
            f for f in os.listdir(VIDEO_DIR)
            if f.lower().endswith((".mp4", ".webm", ".mkv", ".avi", ".mov"))
        ])
        print(f"→ Video directory: {VIDEO_DIR} ({video_count} files)")
    else:
        print(f"→ Video directory not found: {VIDEO_DIR}")
        print("  Source links will show without video playback.")

    print("=" * 55)
    yield
    print("--- Server Shutdown ---")
    app_state.clear()


app = FastAPI(
    title="EduQuery — AI Teaching Assistant API",
    description="Ask questions about your video lectures with streaming responses.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount video directory
if os.path.isdir(VIDEO_DIR):
    app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")


# --- Endpoints ---

@app.get("/")
def read_root():
    return {
        "app": "EduQuery",
        "backend": app_state.get("backend", "unknown"),
        "video_serving": os.path.isdir(VIDEO_DIR),
        "docs": "/docs",
    }


@app.get("/api/videos")
def list_videos():
    """Return a list of available video files."""
    if not os.path.isdir(VIDEO_DIR):
        return {"videos": [], "available": False}

    videos = []
    for f in sorted(os.listdir(VIDEO_DIR)):
        if f.lower().endswith((".mp4", ".webm", ".mkv", ".avi", ".mov")):
            videos.append({"filename": f, "url": f"/videos/{quote(f)}"})

    return {"videos": videos, "available": True}


@app.get("/api/status")
def api_status():
    """Return detailed backend status."""
    status = {
        "backend": app_state.get("backend", "none"),
        "ollama_host": OLLAMA_HOST,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "video_dir": VIDEO_DIR,
        "video_serving": os.path.isdir(VIDEO_DIR),
    }

    if app_state.get("backend") == "pickle" and "df" in app_state:
        status["chunk_count"] = len(app_state["df"])
    elif app_state.get("backend") == "chromadb":
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            collection = client.get_collection(name=CHROMA_COLLECTION)
            status["chunk_count"] = collection.count()
        except Exception:
            status["chunk_count"] = 0

    return status


@app.post("/api/migrate")
def migrate_to_chromadb():
    """Migrate embeddings.pkl data into ChromaDB."""
    added, skipped, err = migrate_pickle_to_chromadb()
    if err:
        raise HTTPException(status_code=500, detail=err)
    return {
        "message": "Migration complete",
        "added": added,
        "skipped": skipped,
    }


@app.post("/ask")
async def handle_ask_question(request: QueryRequest):
    """
    Receives a question, finds relevant video chunks, and returns a streaming response.
    
    Streaming protocol (NDJSON):
    1. {"type": "sources", "sources": [...]}     — sent first with relevant chunks
    2. {"type": "token", "token": "..."}         — one per LLM token
    3. {"type": "done"}                          — signals end of stream
    4. {"type": "error", "message": "..."}       — on error
    """
    print(f"→ Query: {request.question} (backend: {app_state.get('backend', 'unknown')})")

    # Quick small talk check
    def is_quick_small_talk(text: str) -> Optional[str]:
        s = text.strip().lower()
        if len(s.split()) > 3:
            return None
        if re.match(r"^(hi+|hello|hey|hiya)[!.\s]*$", s):
            return "Hi! 👋 How can I help you today? Ask me anything about your video lectures."
        if re.match(r"^(thanks|thank you|thx|ty)[!.\s]*$", s):
            return "You're welcome! Let me know if you have more questions. 😊"
        if re.match(r"^(bye|goodbye|see ya)[!.\s]*$", s):
            return "Goodbye! Come back anytime you need help with your lectures. 👋"
        return None

    quick = is_quick_small_talk(request.question)
    if quick:
        async def quick_gen():
            yield json.dumps({"type": "sources", "sources": []}) + "\n"
            yield json.dumps({"type": "token", "token": quick}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        return StreamingResponse(quick_gen(), media_type="application/x-ndjson")

    # RAG search
    if app_state.get("backend") == "chromadb":
        results, err = search_chromadb(request.question)
    elif "df" in app_state and "all_embeddings" in app_state:
        results, err = search_pickle(request.question, app_state["df"], app_state["all_embeddings"])
    else:
        raise HTTPException(status_code=503, detail="No embeddings backend is loaded. Run read_chunks.py or read_chunks_v2.py first.")

    if err:
        if err == "OLLAMA_NOT_RUNNING":
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Please make sure Ollama is running.")
        raise HTTPException(status_code=500, detail=err)

    if not results:
        raise HTTPException(status_code=404, detail="No matching chunks found for your question.")

    # Build context and sources
    context_text = "\n---\n".join(
        f"Video {r['video_number']} ({r['title']}) at {r['start_time']:.2f}s:\n{r['text']}"
        for r in results
    )

    sources = []
    seen = set()
    for r in results:
        # Deduplicate sources by video + start_time
        key = f"{r['video_number']}_{r['start_time']}"
        if key in seen:
            continue
        seen.add(key)

        source = {
            "video_number": r["video_number"],
            "title": r["title"],
            "start_time": r["start_time"],
            "end_time": r.get("end_time", 0.0),
        }
        video_url = build_video_url(r["video_number"], r["title"], r["start_time"])
        if video_url:
            source["video_url"] = video_url
        sources.append(source)

    # Streaming generator
    async def generate_response():
        # 1. Send sources first
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"

        # 2. Stream LLM tokens
        for token in ask_llm_stream(context_text, request.question, request.history):
            yield json.dumps({"type": "token", "token": token}) + "\n"

        # 3. Signal completion
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")


# --- Run the Server ---
if __name__ == "__main__":
    print(f"Backend: {'ChromaDB' if USE_CHROMADB else 'Pickle'}")
    print(f"Embedding Model: {EMBEDDING_MODEL} | LLM: {LLM_MODEL}")
    print(f"Visit http://{API_HOST}:{API_PORT}/docs for the API playground.")
    uvicorn.run(app, host=API_HOST, port=API_PORT)