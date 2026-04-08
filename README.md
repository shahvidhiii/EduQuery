# EduQuery — Local Video RAG Teaching Assistant

A local, privacy-first AI teaching assistant that extracts knowledge from lecture videos and answers your questions with **clickable source timestamps**. Built with FastAPI, Ollama, and optionally ChromaDB for scalable vector search.

## ✨ Features

- **Video Transcription** — Whisper-based audio extraction + chunking
- **Semantic Search** — Find relevant lecture segments using embedding similarity
- **Streaming LLM Answers** — Real-time streamed responses from local Ollama
- **Clickable Source Timestamps** — Source pills that open an inline video player seeked to the exact moment *(Phase 2)*
- **ChromaDB Integration** — Persistent vector database for large video libraries *(Phase 2)*
- **Conversation History** — Multi-turn context-aware conversations
- **Dark Mode UI** — Glassmorphism design with micro-animations

## Project Structure

```
├── main.py              # FastAPI backend (dual-backend: pickle + ChromaDB)
├── index_v2.html        # Frontend UI (dark theme, clickable sources, video modal)
├── create_chunks.py     # Whisper transcription → JSON chunks
├── read_chunks.py       # Legacy: JSON chunks → embeddings.pkl
├── read_chunks_v2.py    # Phase 2: JSON chunks → ChromaDB collection
├── process_video.py     # Video → audio extraction (ffmpeg)
├── .env                 # Configuration (Ollama, ChromaDB, video paths)
├── requirements.txt     # Python dependencies
├── videos/              # Place your video files here (for clickable sources)
├── audios/              # Extracted audio files
├── jsons/               # Transcribed chunk JSON files
└── chroma_db/           # ChromaDB persistent storage (auto-created)
```

## Quick Setup (Windows / PowerShell)

### 1. Create & activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install --upgrade pip

# PyTorch (CPU-only for Windows):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# All other dependencies:
pip install -r requirements.txt
```

> **Note**: Ensure `ffmpeg` is installed and on PATH for Whisper transcription.

### 3. Configure `.env`

```env
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3
LLM_MODEL=llama3.2
API_PORT=8000

# Phase 2: Enable ChromaDB (set to "true" to switch from pickle)
USE_CHROMADB=false

# Phase 2: Path to video files for clickable timestamp links
VIDEO_DIR=./videos
```

## Workflow

### Standard Pipeline (Pickle Backend)

1. **Extract audio**: Place videos in `videos/`, run `python process_video.py`
2. **Transcribe**: Run `python create_chunks.py` (generates `jsons/`)
3. **Embed**: Run `python read_chunks.py` (generates `embeddings.pkl`)
4. **Start server**: `python main.py`
5. **Open UI**: `http://localhost:5500/index_v2.html` (via `python -m http.server 5500`)

### Phase 2 Pipeline (ChromaDB Backend)

1. Steps 1-2 same as above
2. **Ingest into ChromaDB**: `python read_chunks_v2.py`
   - To re-ingest: `python read_chunks_v2.py --reset`
3. **Enable in `.env`**: Set `USE_CHROMADB=true`
4. **Start server**: `python main.py`

### Clickable Video Sources

1. Place your original video files in the `videos/` folder
2. Set `VIDEO_DIR=./videos` in `.env`
3. Video files are matched by their number (e.g., `01_Introduction.mp4`)
4. Source pills in the chat will show a ▶ indicator and open an inline player

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Server info (backend type, video availability) |
| `POST` | `/ask` | Ask a question (streaming NDJSON response) |
| `GET` | `/api/videos` | List available video files |
| `GET` | `/videos/{filename}` | Serve video files (static) |
| `GET` | `/docs` | Interactive API documentation |

## Troubleshooting

- **"Embeddings database is not loaded"** — Run `read_chunks.py` or `read_chunks_v2.py`
- **"OLLAMA_NOT_RUNNING"** — Start Ollama: `ollama serve`
- **Source pills don't show ▶** — Video files not found in `VIDEO_DIR`
- **ChromaDB not working** — Ensure `USE_CHROMADB=true` in `.env` and run `read_chunks_v2.py`

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **LLM/Embeddings**: Ollama (local, private)
- **Vector DB**: ChromaDB (Phase 2) or Pickle/NumPy (legacy)
- **Transcription**: OpenAI Whisper
- **Frontend**: Vanilla HTML/CSS/JS + marked.js
