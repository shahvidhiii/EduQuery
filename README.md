# EduQuery - Local Video RAG Teaching Assistant

A local, privacy-first teaching assistant that extracts knowledge from lecture videos and answers questions with source timestamps. Built with FastAPI, Ollama, Whisper, and ChromaDB.

## Features

- Video transcription with Whisper
- Semantic search over transcript chunks
- Streaming local LLM answers through Ollama
- Clickable source timestamps for local video playback
- ChromaDB vector storage with a legacy pickle fallback
- Admin page for video processing and chunk inspection

## Project Structure

```text
src/eduquery/          FastAPI backend package
  app.py               API app, RAG search, admin endpoints
  paths.py             Shared project/data path constants
scripts/               Pipeline, migration, diagnostics, and verification scripts
tests/                 API and ChromaDB smoke tests
web/                   Frontend HTML assets
docs/                  Project and migration documentation
data/                  Local runtime data, gitignored
  videos/              Input videos
  audios/              Extracted audio files
  jsons/               Transcribed chunk JSON files
  chroma_db/           ChromaDB persistent storage
  embeddings.pkl       Legacy pickle backend
main.py                Compatibility launcher for src.eduquery.app
process_video.py       Compatibility wrapper for scripts/process_video.py
create_chunks.py       Compatibility wrapper for scripts/create_chunks.py
read_chunks.py         Compatibility wrapper for scripts/read_chunks.py
read_chunks_v2.py      Compatibility wrapper for scripts/read_chunks_v2.py
```

## Quick Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Make sure `ffmpeg` and Ollama are installed and available on PATH.

## Configuration

The default `.env` points runtime data to `data/`:

```env
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3
LLM_MODEL=llama3.2
API_HOST=127.0.0.1
API_PORT=8000
USE_CHROMADB=true
CHROMA_DB_DIR=data/chroma_db
CHROMA_COLLECTION=video_chunks
VIDEO_DIR=data/videos
AUDIO_DIR=data/audios
JSON_DIR=data/jsons
EMBEDDINGS_FILE=data/embeddings.pkl
```

## Workflow

1. Place videos in `data/videos/`.
2. Extract audio: `python process_video.py`
3. Transcribe chunks: `python create_chunks.py`
4. Ingest into ChromaDB: `python read_chunks_v2.py`
5. Start the API: `python main.py`
6. Open the UI through a static server, for example:

```powershell
python -m http.server 5500
```

Then visit `http://localhost:5500/web/index_v2.html`.

The admin dashboard is available from the API at `http://127.0.0.1:8000/admin`.

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Server info |
| `POST` | `/ask` | Ask a question with streaming NDJSON response |
| `GET` | `/api/videos` | List available video files |
| `GET` | `/videos/{filename}` | Serve local video files |
| `GET` | `/admin` | Admin dashboard |
| `GET` | `/docs` | FastAPI interactive docs |

## Notes

- Root-level Python files are compatibility wrappers, so older commands still work.
- Generated media, chunks, embeddings, and ChromaDB files live under `data/` and are ignored by Git.
- See `docs/PROJECT_STRUCTURE.md` for a short layout summary.
