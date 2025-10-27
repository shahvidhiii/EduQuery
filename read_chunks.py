import requests
import os
import json
import pandas as pd
import numpy as np  # <-- NEW IMPORT
from sklearn.metrics.pairwise import cosine_similarity  # <-- NEW IMPORT
import time

# --- Prerequisites ---
# 1. Make sure pandas, numpy, and scikit-learn are installed.
# 2. Make sure Ollama is RUNNING on your computer.
# 3. Make sure you have the embedding model:
#    ollama pull bge-m3
# ---------------------


def create_embedding(text, model="bge-m3"):
    """
    Calls the local Ollama API to create an embedding for a SINGLE text.
    (This is the correct, working function)
    """
    try:
        r = requests.post(
            "http://localhost:11434/api/embeddings", 
            json={
                "model": model,
                "prompt": text
            },
            timeout=60
        )
        r.raise_for_status()
        embedding = r.json().get('embedding')
        if not embedding:
            print(f"Error: 'embedding' key not found for text: {text[:50]}...")
            return None
        return embedding
        
    except requests.exceptions.ConnectionError:
        print("\n--- CRITICAL ERROR: Could not connect to Ollama. ---")
        print("Please make sure Ollama is running in the background.")
        return "OLLAMA_NOT_RUNNING"
    except Exception as e:
        print(f"An error occurred while creating embedding: {e}")
        return None

# --- Main Script ---

# Resolve paths relative to this script so the script works when run from any CWD
base_dir = os.path.dirname(os.path.abspath(__file__))
jsons_folder = os.path.join(base_dir, "jsons")
all_chunks_list = []
chunk_id = 0
embeddings_file = os.path.join(base_dir, "embeddings.pkl")

# --- Check if embeddings file already exists ---
if os.path.exists(embeddings_file):
    print(f"Found existing embeddings file: '{embeddings_file}'. Loading...")
    df = pd.read_pickle(embeddings_file)
    print("Embeddings loaded successfully.")
else:
    print(f"No embeddings file found. Starting embedding creation...")
    
    # --- This is the loop from our previous script ---
    try:
        json_files = os.listdir(jsons_folder)
    except FileNotFoundError:
        print(f"ERROR: The folder '{jsons_folder}' was not found.\nChecked path: {os.path.abspath(jsons_folder)}")
        exit()

    for json_file in json_files:
        if not json_file.endswith(".json"):
            continue
            
        file_path = os.path.join(jsons_folder, json_file)
        
        with open(file_path, "r", encoding="utf-8") as f:
            # FIX #1: 'list_of_chunks' is the JSON content directly
            list_of_chunks = json.load(f) 
        
        print(f"\n--- Processing {json_file} ({len(list_of_chunks)} chunks) ---")
        
        for chunk in list_of_chunks:
            text_to_embed = chunk.get('text')
            if not text_to_embed:
                continue

            # FIX #2: Call the embedding function for each chunk
            embedding = create_embedding(text_to_embed)
            
            if embedding == "OLLAMA_NOT_RUNNING":
                print("Stopping script. Please start Ollama and try again.")
                exit()
            
            if embedding:
                chunk['chunk_id'] = chunk_id
                chunk['embedding'] = embedding
                all_chunks_list.append(chunk)
                chunk_id += 1
            
            if (chunk_id % 20 == 0):
                print(f"Processed {chunk_id} total chunks...")
                
        print(f"--- Finished {json_file} ---")

    print(f"\n--- All {len(json_files)} files processed! ---")
    print(f"Total chunks with embeddings: {len(all_chunks_list)}")

    df = pd.DataFrame.from_records(all_chunks_list)
    df.to_pickle(embeddings_file)
    print(f"DataFrame saved to '{embeddings_file}'.")


# --- (THIS IS YOUR NEW SEARCH LOGIC) ---
# --- Run this part only if the DataFrame was loaded or created successfully ---

if not df.empty:
    print("\n\n--- AI Teaching Assistant Search ---")
    
    while True: # Keep asking for questions
        incoming_query = input("\nAsk a question (or type 'quit' to exit): ")
        
        if incoming_query.lower() == 'quit':
            break

        # 1. Create embedding for the user's question
        #    (We call our *correct* function, not the broken one)
        question_embedding = create_embedding(incoming_query)

        if question_embedding:
            # 2. Get all chunk embeddings from the DataFrame
            #    np.vstack stacks all the lists into a big 2D array
            all_embeddings = np.vstack(df['embedding'].values)
            
            # 3. Calculate similarities
            #    We compare the question's embedding to all chunk embeddings
            similarities = cosine_similarity(all_embeddings, [question_embedding]).flatten()
            
            # 4. Get the top 3 most similar results
            top_results = 3
            # argsort() gets the indices, [::-1] reverses them, [:top_results] gets the top 3
            max_indx = similarities.argsort()[::-1][:top_results] 
            
            print(f"\nTop {top_results} results for your query:")
            
            # 5. Get the matching rows from the DataFrame
            results_df = df.loc[max_indx]
            
            # 6. Print the results nicely
            for index, row in results_df.iterrows():
                print(f"\nSource: {row['title']} (Video {row['number']})")
                print(f"  Time: {row['start']:.2f}s - {row['end']:.2f}s")
                print(f"  Text: {row['text']}")
                print("  ------------------")
        else:
            print("Could not create embedding for the question. Please try again.")
else:
    print("DataFrame is empty. Cannot start search.")