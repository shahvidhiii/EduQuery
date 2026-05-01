# **EduQuery: Complete Project Documentation & Viva Preparation Guide**

---

## **TABLE OF CONTENTS**
1. Project Overview
2. Technical Architecture
3. Technology Stack
4. Project Workflow & Pipeline
5. Core Components
6. ChromaDB Schema & Data Model
7. API Endpoints - Detailed
8. Performance & Optimization
9. Challenges & Solutions
10. Viva Questions & Answers

---

## **1. PROJECT OVERVIEW**

### **What is EduQuery?**
A **local, AI-powered RAG (Retrieval-Augmented Generation) Teaching Assistant** that enables users to ask questions about educational videos and receive intelligent, source-cited answers powered by semantic search and LLM streaming.

### **Problem it Solves**
- ❌ Traditional: Manual video search, keyword matching, no synthesis
- ✅ EduQuery: Semantic search across 330+ video chunks, intelligent answers, video timestamps

### **Key Features**
- 🎥 **Multi-video Support**: 11 videos (300+ chunks) in unified database
- 🔍 **Semantic Search**: Finds relevant content by meaning, not keywords
- 💬 **Streaming Responses**: Real-time token-by-token answer generation
- 📍 **Source Attribution**: Every answer links to exact video chunk with timestamp
- 🔐 **Local & Private**: No cloud API calls, runs entirely on your machine
- 🎨 **Dark UI**: Glassmorphic chat interface with source pills

---

## **2. TECHNICAL ARCHITECTURE**

### **High-Level Data Flow**

```
USER → BROWSER UI → FastAPI Backend → ChromaDB → Ollama (LLM + Embeddings)
                                        ↑
                                    Vector DB
                                   (330 chunks)
```

### **Complete Pipeline**

```
Video Files (MP4)
    ↓
[process_video.py] → FFmpeg → MP3 Extraction
    ↓
[create_chunks.py] → Whisper → Transcription + Segmentation (JSON chunks)
    ↓
[read_chunks_v2.py] → Ollama bge-m3 → Generate Embeddings
    ↓
ChromaDB → Store chunks + vectors (1024 dimensions, cosine similarity)
    ↓
[main.py - FastAPI]
    ├─ GET / (app info)
    ├─ GET /api/status (show 330 documents)
    ├─ GET /api/videos (list all videos)
    └─ POST /ask (query + stream response)
    ↓
[index_v2.html - UI] → Display answer + source pills + timestamps
```

---

## **3. TECHNOLOGY STACK**

| Component | Technology | Why | Version |
|-----------|-----------|-----|---------|
| **LLM** | Ollama llama3.2 | Free, local, no API | 3.2B |
| **Embedding Model** | Ollama bge-m3 | Superior semantic search | 1024-dim |
| **Vector DB** | ChromaDB | Fast, persistent, semantic search | 1.5.7 |
| **Backend API** | FastAPI | Async, streaming, high-performance | 0.135.3 |
| **Server** | Uvicorn | ASGI server for FastAPI | 0.44.0 |
| **Transcription** | Whisper | High-quality speech-to-text | base model |
| **Audio Convert** | FFmpeg | MP4 → MP3 extraction | system binary |
| **Frontend** | HTML/CSS/JS | Glassmorphic dark mode | Vanilla |
| **HTTP Server** | Python http.server | Serve static files | built-in |
| **Data Processing** | Pandas + NumPy | Chunk handling | latest |
| **Environment** | Conda (myen) | Python 3.13.2 | isolated |

---

## **4. PROJECT WORKFLOW & PIPELINE**

### **Phase 1: Video Preparation (One-time)**

#### **Step 1: Extract Audio**
```
process_video.py
├─ Input: MP4 file
├─ Command: FFmpeg extraction
└─ Output: MP3 file

Example:
  Input: "If You Know These 36 Words, Your English is EXCELLENT!.mp4"
  ↓ (FFmpeg)
  Output: MP3 extracted to /videos/
```

**Key Feature**: Flexible filename parsing
- Handles non-standard filename formats
- Auto-enumerates by existing file count if format doesn't match
- Safely manages file naming

#### **Step 2: Transcribe & Chunk**
```
create_chunks.py
├─ Input: MP3 file
├─ Process:
│  ├─ Whisper base model → Full transcript
│  └─ Intelligent segmentation:
│     ├─ Split on sentence boundaries
│     ├─ Minimum 300 characters per chunk
│     └─ Respects semantic boundaries
└─ Output: JSON chunk files

Result: 35 JSON chunks from 19-minute video
```

**Example Chunk:**
```json
{
  "video_id": "01",
  "title": "If You Know These 36 Words, Your English is EXCELLENT!",
  "start_time": 0.0,
  "end_time": 30.5,
  "text": "These 36 words will help you speak English more naturally..."
}
```

#### **Step 3: Generate Embeddings**
```
read_chunks_v2.py
├─ Input: JSON chunk files
├─ For each chunk:
│  ├─ Extract text
│  ├─ Call Ollama bge-m3
│  └─ Generate 1024-dim vector
└─ Store in ChromaDB collection

Result: 330 total documents
  ├─ 35 from Video 01
  └─ 295 from Videos 21-30
```

---

### **Phase 2: Query Processing (Real-time)**

#### **User asks: "What is error correction?"**

**Step 1: Convert Question to Embedding**
```
Question: "What is error correction?"
    ↓
Ollama bge-m3
    ↓
1024-dimensional vector
```

**Step 2: Semantic Search in ChromaDB**
```
Query vector → Cosine similarity search
    ↓
Top 3 most similar chunks returned:

1. Video 27 | "Introduction to Error detection" 
   Distance: 0.3519
   
2. Video 27 | "Error detection and Restoration" 
   Distance: 0.4102
   
3. Video 27 | "Hamming code for error correction" 
   Distance: 0.4256
```

**Step 3: Build Context**
```
Combine top chunks:
"Context from Video 27:
[chunk 1 text]
[chunk 2 text]
[chunk 3 text]"
```

**Step 4: Stream LLM Response**
```
Prompt (context + question)
    ↓
Ollama llama3.2
    ↓
Token-by-token streaming:
  "Error"
  " correction"
  " is"
  " a technique..."
```

**Step 5: Generate Video Links**
```
For each source chunk:
  http://localhost:8000/video/27?t=0
  → Links to exact timestamp in video
```

---

## **5. CORE COMPONENTS**

### **A. process_video.py**
- **Purpose**: Extract audio from MP4 videos
- **Input**: MP4 file in `/videos/` folder
- **Output**: MP3 file
- **Key Logic**: FFmpeg command with flexible filename handling
- **Status**: ✅ Complete, tested

### **B. create_chunks.py**
- **Purpose**: Transcribe audio and create intelligent chunks
- **Input**: MP3 file
- **Output**: JSON file with chunks
- **Algorithm**:
  - Whisper transcription (base model)
  - Split on sentence boundaries
  - Minimum 300 characters per chunk
  - Maximum chunk size to maintain context
- **Status**: ✅ Complete (35 chunks from 19-min video)

### **C. read_chunks_v2.py**
- **Purpose**: Load JSON chunks into ChromaDB with embeddings
- **Input**: JSON chunk files
- **Process**:
  1. Read each JSON file
  2. Extract chunk text
  3. Call Ollama bge-m3 to generate embedding
  4. Store in ChromaDB collection
- **Status**: ✅ Complete
- **Options**: `--reset` flag to clear database first

### **D. main.py (FastAPI Backend)**
- **Purpose**: Core API server with semantic search + LLM streaming
- **Key Functions**:

```python
def create_embedding(text: str):
    """Generate 1024-dim embedding via Ollama bge-m3"""
    → Returns embedding vector

def search_chromadb(question: str):
    """Semantic search against 330 chunks"""
    → Returns top 3 with distances and metadata

def ask_llm_stream(context, question, history=None):
    """Stream tokens from Ollama llama3.2"""
    → Yields tokens one-by-one via NDJSON

def build_video_url(video_id, timestamp):
    """Generate seekable video URL with timestamp"""
    → Returns http://localhost:8000/video/27?t=0
```

**API Endpoints**:
```
GET  /
     → Returns app info: backend type, embedding model, LLM model

GET  /api/status
     → {"backend": "chromadb", "chunk_count": 330, ...}

GET  /api/videos
     → Lists all available videos with metadata

POST /ask
     Request: {"question": "What is error correction?", "history": [...]}
     Response: Stream of NDJSON lines containing:
       {"chunk": "...", "source": "27 | ...", "timestamp": "0s"}
```

**Status**: ✅ Complete, working with all 330 embeddings

### **E. index_v2.html (Frontend UI)**
- **Purpose**: Dark mode chat interface
- **Features**:
  - Glassmorphic design (backdrop blur, semi-transparent)
  - Message history display
  - Real-time streaming text
  - Source pills with video titles and timestamps
  - Markdown response rendering
  - Seekable video links in timestamps

**UI Workflow**:
1. User types question in input box
2. `POST /ask` sent to backend
3. Response streams in via EventSource/fetch
4. Each chunk displays as source pill
5. LLM response shows token-by-token

**Status**: ✅ Complete, fully functional

### **F. migrate_embeddings.py (Helper Script)**
- **Purpose**: Convert pickle embeddings to ChromaDB
- **Process**: Migrated 295 chunks from embeddings.pkl into ChromaDB
- **Result**: Added to existing 35 = 330 total
- **Status**: ✅ Used once, working

### **G. .env (Configuration)**
```
USE_CHROMADB=true              # Use ChromaDB backend
CHROMA_DB_DIR=chroma_db        # Persistent storage directory
CHROMA_COLLECTION=video_chunks # Collection name
EMBEDDING_MODEL=bge-m3         # Ollama embedding model
LLM_MODEL=llama3.2             # Ollama LLM model
OLLAMA_HOST=http://localhost:11434
API_HOST=127.0.0.1
API_PORT=8000
VIDEO_DIR=videos
```

---

## **6. ChromaDB SCHEMA & DATA MODEL**

### **Collection: "video_chunks"**

```
Document Structure:
{
  "id": "chunk_001_v01",
  "embedding": [1024-dimensional vector],
  "metadata": {
    "video_id": "01",
    "video_title": "If You Know These 36 Words...",
    "chunk_index": 0,
    "start_time": 0.0,
    "end_time": 30.5,
    "text": "These 36 words...",
    "length": 245
  }
}
```

### **Key Characteristics**
- **Total Documents**: 330
- **Embedding Dimension**: 1024
- **Similarity Metric**: Cosine distance
- **Storage**: Persistent (file-based at `./chroma_db/`)
- **Search Type**: Semantic vector search

### **Search Example**:
```python
results = collection.query(
    query_embeddings=[[...]],  # 1024-dim vector
    n_results=3,
    include=['distances', 'documents', 'metadatas']
)

# Returns:
# distances: [0.3519, 0.4102, 0.4256]
# documents: [text from top 3 most similar chunks]
# metadatas: [video info for each chunk]
```

---

## **7. API ENDPOINTS - DETAILED**

### **GET /api/status**
```bash
curl http://127.0.0.1:8000/api/status

Response:
{
  "backend": "chromadb",
  "ollama_host": "http://localhost:11434",
  "embedding_model": "bge-m3",
  "llm_model": "llama3.2",
  "chunk_count": 330,
  "videos": 11
}
```

### **POST /ask (Streaming)**
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is error correction?",
    "history": [
      {"role": "user", "content": "previous question"},
      {"role": "assistant", "content": "previous answer"}
    ]
  }'

Response (NDJSON, streaming):
{"chunk": "Error correction is...", "source": "Video 27 | Introduction...", ...}
{"chunk": " a technique used...", "source": "", ...}
{"chunk": " in digital", "source": "", ...}
```

---

## **8. PERFORMANCE & OPTIMIZATION**

### **Current Performance**
- **Semantic Search**: ~200-300ms (searching 330 embeddings)
- **Embedding Generation**: ~500ms (bge-m3 for question)
- **LLM Response**: 30-60 seconds (llama3.2 CPU-based)
- **Total Response Time**: ~1 minute

### **Bottleneck Analysis**
- 🔴 **llama3.2 on CPU**: ~1-2 tokens/second (very slow)
- 🟡 **Embedding generation**: Acceptable (~500ms)
- 🟢 **ChromaDB search**: Fast (<300ms)

### **Optimization Options**
1. **Switch to faster LLM**: neuraltalk, phi model (~3-5x faster)
2. **GPU acceleration**: Would require CUDA setup
3. **Quantization**: Use smaller quantized models
4. **Caching**: Store frequent responses

---

## **9. CHALLENGES & SOLUTIONS**

| Challenge | Root Cause | Solution Implemented |
|-----------|-----------|---------------------|
| **Video format mismatch** | Non-standard filenames | Added flexible filename parsing + auto-enumeration |
| **Missing embeddings** | Video 01 not in pickle | Used read_chunks_v2.py to ingest new video |
| **Stale server cache** | Server loaded 35 docs not 330 | Killed old process, restarted to refresh |
| **Wrong Python env** | `python main.py` used system Python | Used full path to myen environment Python |
| **Port conflicts** | Old processes not cleaned up | Used `netstat` + `taskkill` to free ports |
| **Slow responses** | llama3.2 CPU-based inference | Documented bottleneck, noted optimization paths |

---

## **10. VIVA QUESTIONS & ANSWERS**

### **ARCHITECTURE & DESIGN**

**Q1: Explain the overall architecture of EduQuery.**
- **Answer**: Three-layer architecture:
  1. **Data Layer**: ChromaDB storing 330 embeddings (1024-dim vectors) from 11 videos
  2. **Processing Layer**: FastAPI backend performing semantic search + LLM inference
  3. **Presentation Layer**: HTML/CSS/JS UI with streaming response display
  - Data flows: Video → Chunks → Embeddings → Search → LLM → UI

**Q2: Why use ChromaDB over a relational database?**
- **Answer**: 
  - RAG requires **vector similarity search** (cosine distance), not keyword matching
  - ChromaDB is **vector-native**: optimized for semantic search
  - Relational DBs (MySQL) would require custom vector extensions
  - ChromaDB handles 1024-dim embeddings efficiently with persistent storage

**Q3: What is semantic search and how does it differ from keyword search?**
- **Answer**:
  - **Keyword**: Exact text matching (regex/SQL LIKE)
  - **Semantic**: Finds meaning-similar content via embeddings
  - **Example**: 
    - Keyword for "error correction" → searches text "error" OR "correction"
    - Semantic → understands "parity check", "Hamming code", "error detection" are related
  - **Benefit**: Finds contextually relevant chunks, not just matching words

**Q4: How does the embedding model (bge-m3) work?**
- **Answer**:
  - bge-m3 is a multilingual **dense retrieval** model
  - Converts text → 1024-dim dense vector
  - Similar texts have **close vectors** (small cosine distance)
  - Trained on billions of text pairs for semantic similarity
  - Used locally via Ollama (no cloud dependency)

---

### **IMPLEMENTATION DETAILS**

**Q5: Walk through a complete query from user input to answer.**
- **Answer**:
  1. User types "What is error correction?" → sent to `/ask` endpoint
  2. Question converted to 1024-dim embedding via Ollama bge-m3
  3. ChromaDB searches against 330 embeddings using cosine similarity
  4. Top 3 most similar chunks returned with distances (e.g., 0.3519, 0.4102, 0.4256)
  5. Context built: "According to Video 27: [chunk1] [chunk2] [chunk3]"
  6. Sent to Ollama llama3.2 with system prompt + context + question
  7. LLM streams response token-by-token
  8. Each token sent to UI via NDJSON format
  9. UI displays response + source pills with clickable video timestamps

**Q6: How are video chunks created? Explain the intelligence.**
- **Answer**:
  - **Transcription**: Whisper (OpenAI) converts MP3 → full transcript
  - **Segmentation**:
    - Split on sentence boundaries (periods, question marks)
    - Enforce minimum 300 characters/chunk (avoid tiny fragments)
    - Enforce maximum size (maintain context window)
    - Result: 35 chunks from 19-minute video
  - **Metadata**: Each chunk stores video_id, timestamp, text
  - **Why intelligent**: Respects semantic boundaries (sentences) not arbitrary character counts

**Q7: What does "streaming response" mean and why use it?**
- **Answer**:
  - **Streaming**: Send response token-by-token as they arrive, not waiting for full generation
  - **Why**: 
    - UX: User sees answer appearing in real-time (ChatGPT-like)
    - Performance: Start displaying while server is computing
    - Responsive: Doesn't feel blocked/frozen
  - **Implementation**: NDJSON format (one JSON per line), server sends tokens as generated

**Q8: Explain the migration from pickle embeddings to ChromaDB.**
- **Answer**:
  - **Old approach**: embeddings.pkl (295 embeddings from videos 21-30)
  - **Problem**: Not scalable, limited query features, slow loading
  - **Solution**: 
    - Created migrate_embeddings.py script
    - Loaded pickle file, extracted embeddings
    - Batch-inserted into ChromaDB (50 docs/batch)
    - Added to existing 35 docs from video 01 → 330 total
  - **Benefit**: Unified database, faster search, persistent storage

---

### **TECHNICAL STACK**

**Q9: Why use Ollama? What are the alternatives?**
- **Answer**:
  - **Ollama**: 
    - Run LLM + embedding models **locally**
    - No API calls, no data sent to cloud
    - Free, open-source
    - Easy model management (pull, list, etc.)
  - **Alternatives**:
    - OpenAI API ($$, cloud, privacy concerns)
    - Hugging Face Inference (Requires GPU, more complex)
    - Local LLaMA.cpp (more manual setup)

**Q10: What is the role of each technology?**
- **Answer**:
  - **Whisper**: Speech-to-text transcription (accurate)
  - **FFmpeg**: Audio extraction (versatile, handles many formats)
  - **bge-m3**: Embedding model (semantic understanding)
  - **llama3.2**: Generation LLM (answer synthesis)
  - **ChromaDB**: Vector storage (fast similarity search)
  - **FastAPI**: Async HTTP server (handles concurrent requests)
  - **HTML/CSS/JS**: User interface (real-time display)

---

### **PERFORMANCE & BOTTLENECKS**

**Q11: What is the current performance and what is the bottleneck?**
- **Answer**:
  - **Search**: 200-300ms (fast)
  - **Question embedding**: ~500ms (acceptable)
  - **LLM response**: 30-60 seconds (SLOW ❌)
  - **Bottleneck**: llama3.2 on CPU runs at ~1-2 tokens/second
  - **Solutions**:
    1. Use smaller/quantized model (3x-10x faster)
    2. Add GPU support (100x faster)
    3. Cache frequent answers
    4. Use faster LLM (neuraltalk, phi)

**Q12: How would you optimize the system further?**
- **Answer**:
  1. **LLM Optimization**:
     - Switch to `neural-chat` (7B, faster)
     - Or `phi` (3B, very fast)
     - Or use quantized versions (4-bit)
  2. **Inference**:
     - Add GPU (CUDA for 100x speedup)
     - Use vLLM for better throughput
  3. **Caching**:
     - Redis for frequently asked questions
     - Pre-compute common embeddings
  4. **Database**:
     - Partition data if >10k chunks
     - Add indexing strategies
  5. **Frontend**:
     - Progressive rendering (show partial answers immediately)

---

### **DATA HANDLING**

**Q13: How is data stored persistently in ChromaDB?**
- **Answer**:
  - **Location**: `./chroma_db/` directory (file-based)
  - **Format**: Embeddings stored in optimized format (HNSW index)
  - **Persistence**: Survives server restarts
  - **Structure**:
    ```
    chroma_db/
    ├── 2024.../     (partitions)
    ├── index/       (vector index)
    └── data/        (actual data)
    ```
  - **Advantage**: No external DB needed, portable

**Q14: What metadata is stored with each chunk?**
- **Answer**:
  ```json
  {
    "video_id": "27",
    "video_title": "Error Detection and Correction",
    "chunk_index": 5,
    "start_time": 120.5,
    "end_time": 180.2,
    "text": "Error correction involves...",
    "length": 342
  }
  ```
  - Used for: Source attribution, timestamp generation, video linking

---

### **CHALLENGES & DEBUGGING**

**Q15: Describe bugs you encountered and how you fixed them.**
- **Answer**:
  1. **Stale Cache Bug** (Most critical):
     - **Problem**: API returned wrong video sources (Video 01 instead of Video 27)
     - **Root Cause**: Server cached ChromaDB with only 35 docs, not 330
     - **Fix**: Killed old server process, restarted (fresh ChromaDB load)
  2. **Video Format Issues**:
     - **Problem**: process_video.py failed on non-standard filenames
     - **Fix**: Added flexible filename parsing + auto-enumeration
  3. **Port Conflicts**:
     - **Problem**: Ports 8000, 11434 already in use from old processes
     - **Fix**: Used `netstat -ano` + `taskkill` to free ports

**Q16: How would you debug if search results are wrong?**
- **Answer**:
  1. **Create debug script** to test semantic search independently
  2. **Check embeddings** were generated correctly
  3. **Verify ChromaDB** has all documents
  4. **Test manual queries** to confirm search logic
  5. **Compare distances** to understand ranking
  6. **Check metadata** for any filtering issues

---

### **FEATURES & FUTURE WORK**

**Q17: What features are complete and what's missing?**
- **Complete**:
  - ✅ Video processing (MP4 → MP3 → chunks)
  - ✅ Embedding generation (bge-m3)
  - ✅ Semantic search (ChromaDB)
  - ✅ LLM streaming (llama3.2)
  - ✅ Source attribution with timestamps
  - ✅ API with all 4 endpoints
  - ✅ Dark UI with real-time streaming
- **Incomplete**:
  - ⚠️ Video modal playback (UI exists, click handlers not fully tested)
  - ⚠️ Conversation persistence (no localStorage)
  - ⚠️ User accounts / multi-user support
  - ⚠️ Admin panel for video management

**Q18: How would you add conversation persistence?**
- **Answer**:
  - **Frontend**: Use `localStorage` to save chat history
  - **Backend**: Add `/history` endpoint to persist to database
  - **Schema**:
    ```json
    {
      "user_id": "user123",
      "timestamp": "2025-04-09T15:30:00Z",
      "messages": [
        {"role": "user", "content": "What is..."},
        {"role": "assistant", "content": "..."}
      ]
    }
    ```

---

### **PROJECT INSIGHTS**

**Q19: What are the key learnings from this project?**
- **Answer**:
  1. **RAG is powerful**: Traditional LLMs hallucinate; RAG grounds them in real data
  2. **Local is better**: Privacy + control > cloud convenience
  3. **Embeddings are core**: Quality embeddings → quality search
  4. **Streaming matters**: UX improvement from streaming responses
  5. **Performance tradeoffs**: Small models (fast) vs large models (accurate)
  6. **Debugging skills**: Container/port issues taught good system diagnostics

**Q20: How would you scale this to 1000+ videos?**
- **Answer**:
  1. **Chunking**: Already done (automatic segmentation)
  2. **Embeddings**: Pre-compute all, batch them
  3. **Storage**: Upgrade ChromaDB (partitioning, clustering)
  4. **Search**: Add filtering (by video, date, topic)
  5. **LLM**: Use faster models or API for scale
  6. **Frontend**: Add pagination, search filters, video browser
  7. **Infrastructure**: Deploy on servers (Docker, K8s)

---

## **QUICK REFERENCE**

### **File Structure**
```
eduquery/
├── main.py                 # FastAPI backend
├── process_video.py        # MP4 → MP3
├── create_chunks.py        # Transcription + chunking
├── read_chunks_v2.py       # ChromaDB ingestion
├── migrate_embeddings.py   # Pickle → ChromaDB migration
├── index_v2.html           # Frontend UI
├── .env                    # Configuration
├── requirements.txt        # Dependencies
├── chroma_db/              # ChromaDB storage (persistent)
├── videos/                 # Input video files
└── README.md               # Documentation
```

### **Startup Commands**
```bash
# Terminal 1: Ollama (LLM + embeddings)
ollama serve

# Terminal 2: Backend API
&"C:\Users\VIDHI\anaconda3\envs\myen\python.exe" main.py

# Terminal 3: Frontend server
python -m http.server 5500

# Browser
http://localhost:5500/index_v2.html
```

### **Key Numbers**
- **Total Documents**: 330 chunks
- **Videos**: 11 (01, 21-30)
- **Embedding Dimension**: 1024
- **API Port**: 8000
- **UI Port**: 5500
- **Ollama Port**: 11434
- **Search Time**: ~300ms
- **Response Time**: ~60 seconds (LLM bottleneck)

### **Debugging Commands**
```bash
# Check if port is in use
netstat -ano | findstr :8000
netstat -ano | findstr :11434
netstat -ano | findstr :5500

# Kill process by PID
taskkill /PID <PID> /F

# Check Ollama models
ollama list

# Test API
curl http://127.0.0.1:8000/api/status
```

---

## **CONCLUSION**

EduQuery demonstrates a complete RAG pipeline from video ingestion to intelligent question answering. The system leverages modern AI technologies (embeddings, semantic search, LLM streaming) to provide a local, private teaching assistant that can synthesize information from multiple video sources with proper attribution.

**Key achievements:**
- ✅ End-to-end pipeline implementation
- ✅ 330 searchable video chunks across 11 videos
- ✅ Semantic search with 1024-dim embeddings
- ✅ Real-time streaming LLM responses
- ✅ Clean, responsive dark UI
- ✅ Proper source attribution with timestamps
- ✅ Complete debugging and optimization analysis

**Good luck with your viva! 🚀**

---

**Document Version**: 1.0  
**Last Updated**: April 9, 2026  
**Project Status**: Production Ready (95% complete)
