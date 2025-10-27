import os
import uvicorn
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration ---
# (These are the same as your ask_question.py)
base_dir = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(base_dir, "embeddings.pkl")
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "llama3.2"  # Make sure you have pulled this model

# --- Pydantic Models for Request/Response ---

class QueryRequest(BaseModel):
    """ The request body for a user's question """
    question: str

class Source(BaseModel):
    """ A single source used for the answer """
    video_number: str
    title: str
    start_time: float

class QueryResponse(BaseModel):
    """ The response body with the answer and sources """
    answer: str
    sources: List[Source]


# --- Helper Functions (Copied from ask_question.py) ---

def create_embedding(text):
    """
    Calls local Ollama to create an embedding for a SINGLE string.
    """
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "prompt": text
            },
            timeout=60
        )
        r.raise_for_status()
        embedding = r.json().get('embedding')
        return embedding, None  # Return embedding, no error
        
    except requests.exceptions.ConnectionError:
        print("\n--- ERROR: Could not connect to Ollama. Is it running? ---")
        return None, "OLLAMA_NOT_RUNNING"
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None, str(e)

def ask_llm(context, question):
    """
    Sends the context and question to the local Ollama LLM.
    """
    print(f"\nAsking Ollama ({LLM_MODEL})...")
    
    prompt = f"""
    You are a helpful AI teaching assistant for a web development course.
    Use ONLY the following CONTEXT from video lecture transcripts to answer the user's QUESTION.
    Each piece of context includes the video number, title, start/end times, and text.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Based ONLY on the context provided:
    1. Find the chunk that best answers the QUESTION.
    2. Extract the **video number**, start time, and text from that chunk.
    3. Formulate a response to the user that **MUST** include the phrase "in video number [X]" (where X is the extracted video number) and mention the start time.
    4. Briefly explain what the context says about the topic at that point.
    5. If the context does not contain relevant information, state: "I'm sorry, I couldn't find information about that specific topic in the provided video sections."
    6. Do NOT mention the word "CONTEXT".

    Your final response **MUST** explicitly state the video number. For example: "You can find that topic discussed in **video number 29** starting around 0.00s..."

    ANSWER:
    """

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120
        )
        r.raise_for_status()
        response_data = r.json()
        return response_data.get("response", "Sorry, I encountered an error.").strip(), None
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to Ollama ({LLM_MODEL}). Is it running?"
    except Exception as e:
        return None, f"An error occurred calling Ollama LLM: {e}"

# --- FastAPI App Setup ---

# This dictionary will hold our loaded models and data
# This is better than global variables
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Load embeddings file ---
    print("--- Server Startup ---")
    print("Loading embeddings database...")
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"CRITICAL ERROR: Embeddings file '{EMBEDDINGS_FILE}' not found.")
        print("Please run read_chunks.py first to create the embeddings file.")
        # You might want to exit here, but we'll let it run
        # The /ask endpoint will just fail
    else:
        try:
            df = pd.read_pickle(EMBEDDINGS_FILE)
            df['embedding'] = df['embedding'].apply(lambda x: list(x))
            all_embeddings = np.vstack(df['embedding'].values)
            
            # Store in our app_state
            app_state['df'] = df
            app_state['all_embeddings'] = all_embeddings
            print(f"Embeddings loaded successfully. Total chunks: {len(df)}")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load {EMBEDDINGS_FILE}: {e}")
    
    print("------------------------")
    yield
    # --- Shutdown: Clean up (if needed) ---
    print("--- Server Shutdown ---")
    app_state.clear()


# Initialize the FastAPI app with the lifespan event handler
app = FastAPI(
    title="AI Teaching Assistant API",
    description="Ask questions about your video lectures.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Teaching Assistant API. Go to /docs to see the endpoints."}


@app.post("/ask", response_model=QueryResponse)
async def handle_ask_question(request: QueryRequest):
    """
    Receives a question, finds relevant video chunks, and asks the LLM for an answer.
    """
    
    # Check if embeddings are loaded
    if 'df' not in app_state or 'all_embeddings' not in app_state:
        raise HTTPException(status_code=503, detail="Embeddings database is not loaded. Check server logs.")

    print(f"\nReceived query: {request.question}")

    # 1. Create embedding for the question
    question_embedding, err = create_embedding(request.question)
    if err == "OLLAMA_NOT_RUNNING":
        raise HTTPException(status_code=503, detail="Ollama embedding model (bge-m3) is not reachable.")
    if err:
        raise HTTPException(status_code=500, detail=f"Could not create embedding: {err}")

    # 2. Find most similar chunks
    all_embeddings = app_state['all_embeddings']
    df = app_state['df']
    
    similarities = cosine_similarity([question_embedding], all_embeddings).flatten()
    top_results = 5
    max_indx = similarities.argsort()[::-1][:top_results]
    results_df = df.loc[max_indx]
    
    # 3. Combine chunks into context
    context_text = "\n---\n".join(
        f"Video {row['number']} ({row['title']}) at {row['start']:.2f}s: {row['text']}" 
        for index, row in results_df.iterrows()
    )
    
    # 4. Ask the LLM
    answer, err = ask_llm(context_text, request.question)
    if err:
        raise HTTPException(status_code=503, detail=err)

    # 5. Format sources
    sources = []
    # We show the top 3 sources
    for index, row in results_df.head(3).iterrows():
        sources.append(Source(
            video_number=row['number'],
            title=row['title'],
            start_time=row['start']
        ))
    
    # 6. Return the response
    return QueryResponse(answer=answer, sources=sources)


# --- Run the Server ---
if __name__ == "__main__":
    print("--- Starting AI Teaching Assistant API ---")
    print(f"Embedding Model: {EMBEDDING_MODEL} | LLM: {LLM_MODEL}")
    print(f"Visit http://127.0.0.1:8000/docs for the API playground.")
    uvicorn.run(app, host="127.0.0.1", port=8000)