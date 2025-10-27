import whisper
import os
import json
import torch

# --- 1. Setup ---
print("Checking for device...")
device = "cpu" 
print(f"Using device: {device}")

model_name = "base"
# Use absolute paths based on this script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
audio_folder = os.path.join(base_dir, "audios")
json_folder = os.path.join(base_dir, "jsons")
os.makedirs(json_folder, exist_ok=True)

MIN_CHUNK_CHARS = 300 

# --- 2. Load Model (Only Once) ---
print(f"Loading the '{model_name}' model (this may take a moment)...")
try:
    model = whisper.load_model(model_name, device=device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 3. Start Processing Loop ---
print(f"\nStarting transcription check for all files in '{audio_folder}'...") # <-- Text changed slightly

try:
    audio_files = os.listdir(audio_folder)
except FileNotFoundError:
    print(f"ERROR: The folder '{audio_folder}' was not found.\nChecked path: {os.path.abspath(audio_folder)}")
    exit()

for audio_file in audio_files:
    if not os.path.isfile(os.path.join(audio_folder, audio_file)) or not audio_file.endswith(".mp3"):
        continue

    # --- (NEW!) CHECK IF JSON ALREADY EXISTS ---
    # We calculate the *expected* output path first
    base_name = os.path.splitext(audio_file)[0] 
    json_filename = f"{base_name}.json"
    output_path = os.path.join(json_folder, json_filename)

    # If the file exists, skip this loop
    if os.path.exists(output_path): # <-- NEW
        print(f"-> Skipping '{audio_file}': JSON already exists.") # <-- NEW
        continue # <-- NEW (skips to the next audio_file)

    # --- If we are here, the JSON does not exist, so we process the file ---
    
    print(f"\nProcessing '{audio_file}'...")
    input_path = os.path.join(audio_folder, audio_file)

    try:
        # --- 4. Get Number and Title ---
        # (base_name is already calculated above)
        number = base_name.split("_")[0]
        title = base_name.split("_")[1]

        # --- 5. Transcribe File ---
        result = model.transcribe(
            input_path, 
            language="en",
            fp16=False
        )

        # --- 6. Combine Segments into Larger Chunks ---
        print("-> Combining small segments into larger chunks...")
        
        combined_chunks = []
        temp_text = ""
        temp_start = 0.0
        
        for i, segment in enumerate(result["segments"]):
            
            if not temp_text:
                temp_start = segment["start"]

            temp_text += " " + segment["text"]
            
            if len(temp_text) > MIN_CHUNK_CHARS or i == len(result["segments"]) - 1:
                
                combined_chunks.append({
                    "number": number,
                    "title": title,
                    "start": temp_start,
                    "end": segment["end"], 
                    "text": temp_text.strip()
                })
                
                temp_text = ""
        
        print(f"-> Created {len(combined_chunks)} combined chunks (from {len(result['segments'])} original segments).")
        
        # --- 7. Save JSON File ---
        # (output_path is already calculated above)
        
        # Save the *new* combined_chunks list
        with open(output_path, "w") as f:
            json.dump(combined_chunks, f, indent=4) 
        
        print(f"-> Successfully transcribed and saved to '{output_path}'")

    except Exception as e:
        print(f"-> ERROR processing '{audio_file}': {e}")

print("\n--- All files processed! ---")