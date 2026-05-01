import os
import subprocess
import sys
import json

# Define the source and destination folders
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
video_folder = os.getenv("VIDEO_DIR", os.path.join(data_dir, "videos"))
audio_folder = os.getenv("AUDIO_DIR", os.path.join(data_dir, "audios"))
json_folder = os.getenv("JSON_DIR", os.path.join(data_dir, "jsons"))
registry_file = os.path.join(data_dir, "video_registry.json")

# Create the 'audios' folder if it doesn't already exist
os.makedirs(audio_folder, exist_ok=True)

def load_registry():
    if os.path.exists(registry_file):
        with open(registry_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(registry_file, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=4)

def get_video_number(video_filename):
    """
    Get or assign a persistent video number for a filename.
    """
    registry = load_registry()
    
    # If already in registry, return it
    if video_filename in registry:
        return registry[video_filename]
    
    # Otherwise, assign the next available number
    existing_numbers = [int(n) for n in registry.values() if n.isdigit()]
    # Also check existing files in audios/jsons to avoid collisions if registry was lost
    for folder in [audio_folder, json_folder]:
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                parts = f.split("_", 1)
                if len(parts) >= 2 and parts[0].isdigit():
                    existing_numbers.append(int(parts[0]))
    
    next_num = 1
    if existing_numbers:
        next_num = max(existing_numbers) + 1
    
    video_number = str(next_num).zfill(2)
    registry[video_filename] = video_number
    save_registry(registry)
    return video_number

def get_existing_audio_for_video(video_filename):
    """
    Check if an audio file already exists for this video.
    """
    registry = load_registry()
    video_num = registry.get(video_filename)
    if not video_num:
        return None
        
    for audio_file in os.listdir(audio_folder):
        if audio_file.startswith(f"{video_num}_") and audio_file.lower().endswith(".mp3"):
            return audio_file
    return None

def process_single_video(video_filename):
    """
    Process a single video file: extract audio via ffmpeg.
    """
    if not os.path.isfile(os.path.join(video_folder, video_filename)):
        return False, f"Video '{video_filename}' not found in {video_folder}"

    # Get or assign number
    video_number = get_video_number(video_filename)
    
    # Check if already processed
    existing = get_existing_audio_for_video(video_filename)
    if existing:
        return True, f"Skipped '{video_filename}': audio already exists as '{existing}'"

    base_name = os.path.splitext(video_filename)[0][:40]
    input_path = os.path.join(video_folder, video_filename)
    output_filename = f"{video_number}_{base_name}.mp3"
    output_path = os.path.join(audio_folder, output_filename)

    try:
        print(f"Converting '{video_filename}' -> '{output_filename}'...")
        subprocess.run([
            "ffmpeg",
            "-i", input_path,
            "-vn",
            "-q:a", "0",
            output_path
        ], check=True)
        return True, f"Successfully converted '{video_filename}' to '{output_filename}'"
    except Exception as e:
        return False, f"Error processing '{video_filename}': {e}"

def process_all_videos():
    """Process all video files in the videos folder."""
    files = [
        f for f in os.listdir(video_folder)
        if os.path.isfile(os.path.join(video_folder, f))
        and f.lower().endswith((".mp4", ".webm", ".mkv", ".avi", ".mov"))
    ]

    if not files:
        print("No video files found.")
        return

    # Use alphabetical order ONLY for initial processing of multiple new files
    # to keep them somewhat orderly, but registry will make it permanent.
    for video_file in sorted(files):
        success, message = process_single_video(video_file)
        print(f"  {'[OK]' if success else '[FAIL]'} {message}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        success, msg = process_single_video(target)
        print(msg)
    else:
        process_all_videos()
