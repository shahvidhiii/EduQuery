# Local-Video-RAG — AI Teaching Assistant (Local Video RAG)

A small project that extracts chunks from lecture videos, builds embeddings, and provides a local web UI to ask questions about the video content. The backend is a FastAPI server that finds relevant chunks and asks a local LLM (Ollama) for a formulated answer. This repo includes a polished frontend `index_v2.html` you can preview locally.

## What you'll find

- `main.py` — FastAPI app exposing `/ask` that returns answers and source timestamps.
- `ask_question.py`, `read_chunks.py`, `create_chunks.py` — helper scripts for embedding creation / querying.
- `index.html` — original frontend (kept for compatibility).
- `index_v2.html` — improved frontend UI (recommended to preview).
- `requirements.txt` — Python dependencies (see notes below).

## Quick setup (Windows / PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

Notes about PyTorch: PyTorch wheels are platform-specific. For a CPU-only install on Windows you can run the PyTorch recommended command from the official site. The `requirements.txt` includes `torch` as a placeholder; if you want a quick CPU install run:

```powershell
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining requirements:

```powershell
pip install -r requirements.txt
```

If you run into issues with `openai-whisper`, ensure you have `ffmpeg` installed on your system (add it to PATH) and that `torch` is installed first.

## Running the backend (FastAPI)

From the repository root (with the environment active):

```powershell
# start the FastAPI app
python -m uvicorn main:app --reload
```

The app will run on `http://127.0.0.1:8000` by default. Check `http://127.0.0.1:8000/docs` for the automatic API docs.

Important:
- The server expects an embeddings pickle file at `embeddings.pkl` (see `read_chunks.py` / `create_chunks.py`). If embeddings are not present, the `/ask` endpoint will return 503.
- This project expects a local Ollama service for embeddings/LLM calls. Ensure Ollama is running if you rely on it. Replace or adapt `ask_llm` / `create_embedding` in `main.py` if you use a different LLM endpoint.

## Preview the frontend

Option A (recommended) — serve via Python HTTP server (allows fetch to `localhost`):

```powershell
# from repo root
python -m http.server 5500
# open http://localhost:5500/index_v2.html in your browser
```

Option B — open the file directly in your browser (may experience fetch restrictions in some browsers):

- Double-click `index_v2.html` or open it with your browser.

The page posts to `http://127.0.0.1:8000/ask` — ensure the FastAPI server above is running.

## Typical workflow

1. Use `create_chunks.py` to transcribe and chunk videos (requires `openai-whisper` + ffmpeg). This will generate JSONs or chunk files.
2. Use `read_chunks.py` to convert chunks into an embeddings DataFrame (and save `embeddings.pkl`).
3. Start the FastAPI app (`main.py`).
4. Open `index_v2.html` and ask questions.

## Notes and troubleshooting

- If you get JSON shape / key errors for sources on the frontend, the frontend is tolerant, but check `main.py`'s `QueryResponse` model and the data produced by `read_chunks.py`.
- If whisper/torch fail to import or install, ensure you have a compatible Python version (3.10+ recommended) and install `torch` first.
- If you prefer to use the original `index.html`, it's still present. I created `index_v2.html` as a polished UI — I can replace `index.html` with this version if you prefer.

## Next steps I can help with

- Replace `index.html` with the new UI and add a small config panel to change the API URL.
- Add simple sanitization for bot HTML output or an allow-list of tags.
- Add a small script to build `embeddings.pkl` automatically and a short test to validate the stack.

If you want me to overwrite `index.html` with the new frontend or pin exact dependency versions for your environment (CUDA vs CPU), tell me which target (CPU/CUDA) and I'll update `requirements.txt` and the README accordingly.
