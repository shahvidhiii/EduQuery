import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd # To read pickle file

# --- Configuration ---
# Resolve paths relative to this script so it works when launched from any CWD
base_dir = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(base_dir, "embeddings.pkl") # Load the embeddings file
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"  # Local model for embeddings
LLM_MODEL = "llama3.2"    # Local model for answering (Make sure you have pulled this!)

# --- API Functions (Using Ollama) ---

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
        return embedding
        
    except requests.exceptions.ConnectionError:
        print("\n--- ERROR: Could not connect to Ollama. Is it running? ---")
        return "OLLAMA_NOT_RUNNING"
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

def ask_llm(context, question):
    """
    Sends the context and question to the local Ollama LLM.
    """
    print(f"\nAsking Ollama ({LLM_MODEL})...")
    
    # Using the stricter prompt from our conversation
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
                # Optional: Add parameters like temperature if needed
                # "options": {
                #     "temperature": 0.3
                # }
            },
            timeout=120 # Increase timeout for local LLM
        )
        r.raise_for_status()
        response_data = r.json()
        return response_data.get("response", "Sorry, I encountered an error.").strip()

    except requests.exceptions.ConnectionError:
         print(f"\n--- ERROR: Could not connect to Ollama ({LLM_MODEL}). Is it running? ---")
         return None
    except Exception as e:
        print(f"An error occurred calling Ollama LLM: {e}")
        return None

# --- Main Search Logic ---
def main():
    # Load the DataFrame with embeddings
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Error: Embeddings file '{EMBEDDINGS_FILE}' not found.")
        print("Please run the script that creates embeddings first (e.g., read_chunks.py).")
        return

    print("Loading embeddings database...")
    try:
        df = pd.read_pickle(EMBEDDINGS_FILE)
        df['embedding'] = df['embedding'].apply(lambda x: list(x))
        all_embeddings = np.vstack(df['embedding'].values)
        print("Embeddings loaded successfully.")
    except Exception as e:
        print(f"Error loading {EMBEDDINGS_FILE}: {e}")
        return

    print("\n\n--- AI Teaching Assistant (Local Ollama) ---")
    print(f"Embedding Model: {EMBEDDING_MODEL} | LLM: {LLM_MODEL} | Chunks: {len(df)}")
    
    while True:
        incoming_query = input("\nAsk a question (or type 'quit' to exit): ")
        if incoming_query.lower() == 'quit':
            break

        # 1. Create embedding for the question (uses local Ollama)
        question_embedding = create_embedding(incoming_query)

        if question_embedding == "OLLAMA_NOT_RUNNING":
             print("Exiting.")
             break # Stop if Ollama isn't running
        if not question_embedding:
            print("Could not create embedding for the question.")
            continue

        # 2. Find most similar chunks
        similarities = cosine_similarity([question_embedding], all_embeddings).flatten()
        top_results = 5 # Using 5 for better context
        max_indx = similarities.argsort()[::-1][:top_results]
        results_df = df.loc[max_indx]
        
        # 3. Combine chunks into one context string
        context_text = "\n---\n".join(
            f"Video {row['number']} ({row['title']}) at {row['start']:.2f}s: {row['text']}" 
            for index, row in results_df.iterrows()
        )
        
        # 4. Ask the LLM (uses local Ollama)
        answer = ask_llm(context_text, incoming_query)
        
        if answer:
            print("\n--- AI Answer ---")
            print(answer)
            print("\n--- Sources Used (Top 3) ---")
            # Show only top 3 sources for brevity
            for index, row in results_df.head(3).iterrows():
                print(f"  - Video {row['number']}: {row['title']} (Time: {row['start']:.2f}s)")
        else:
            print("Sorry, I couldn't get an answer from the local LLM.")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()