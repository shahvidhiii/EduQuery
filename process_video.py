import os
import subprocess

# Define the source and destination folders
video_folder = "videos"
audio_folder = "audios"

# Create the 'audios' folder if it doesn't already exist
os.makedirs(audio_folder, exist_ok=True)

# Get the list of files from the 'videos' folder
files = os.listdir(video_folder)

print("Starting conversion...")

for file in files:
    # Make sure we are only processing files, not folders
    if not os.path.isfile(os.path.join(video_folder, file)):
        continue
    
    try:
        # Safely split the filename to get the parts you need
        # os.path.splitext removes the extension (e.g., .mp4)
        base_name = os.path.splitext(file)[0]
        
        # This parsing assumes the format is like "tut-01_My File Name.mp4"
        tutorial_number = base_name.split("_")[0].split("-")[1]
        file_name = base_name.split("_ ")[1]
        
        # Define the full path for the input and output files
        input_path = os.path.join(video_folder, file)
        output_filename = f"{tutorial_number}_{file_name}.mp3"
        output_path = os.path.join(audio_folder, output_filename)
        
        print(f"Converting '{file}' to '{output_filename}'...")

        # Construct and run the ffmpeg command
        subprocess.run([
            "ffmpeg",
            "-i", input_path,  # Input file
            "-vn",             # No video: tells ffmpeg to ignore the video stream
            "-q:a", "0",       # Best available variable audio quality
            output_path        # Output file
        ], check=True) # check=True will show an error if ffmpeg fails

    except IndexError:
        print(f"-> Skipping '{file}': Filename does not match the expected format.")
    except Exception as e:
        print(f"-> An error occurred with '{file}': {e}")

print("\nConversion complete! ✨")