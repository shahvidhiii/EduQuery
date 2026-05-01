"""
cleanup_duplicates.py — Remove duplicate audio/json files, keeping only
the first (lowest-numbered) copy of each unique video.
Also cleans up duplicate ChromaDB entries.
"""

import os
import json
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
audio_folder = os.getenv("AUDIO_DIR", os.path.join(data_dir, "audios"))
json_folder = os.getenv("JSON_DIR", os.path.join(data_dir, "jsons"))
video_folder = os.getenv("VIDEO_DIR", os.path.join(data_dir, "videos"))


def normalize_name(filename):
    """Strip number prefix and extension, normalize for comparison."""
    base = os.path.splitext(filename)[0]
    # Remove the number prefix like "01_", "02_", etc.
    parts = base.split("_", 1)
    if len(parts) >= 2 and parts[0].isdigit():
        name_part = parts[1]
    else:
        name_part = base
    return name_part.lower().replace(" ", "").replace("-", "").replace("_", "")


def find_duplicates(folder, extension):
    """
    Group files by their normalized name.
    Returns dict: { normalized_name: [list of filenames sorted by number] }
    """
    groups = {}
    files = [f for f in os.listdir(folder) if f.lower().endswith(extension)]
    
    for f in sorted(files):
        norm = normalize_name(f)
        if norm not in groups:
            groups[norm] = []
        groups[norm].append(f)
    
    return groups


def cleanup_folder(folder, extension, dry_run=False):
    """Remove duplicate files, keeping only the first (lowest-numbered) copy."""
    if not os.path.isdir(folder):
        print(f"  Folder not found: {folder}")
        return 0

    groups = find_duplicates(folder, extension)
    removed = 0

    for norm_name, files in groups.items():
        if len(files) <= 1:
            continue  # No duplicates

        keep = files[0]  # Keep the first one (lowest number)
        duplicates = files[1:]

        print(f"  Keeping : {keep}")
        for dup in duplicates:
            filepath = os.path.join(folder, dup)
            if dry_run:
                print(f"  [DRY] Would delete: {dup}")
            else:
                os.remove(filepath)
                print(f"  Deleted : {dup}")
            removed += 1

    return removed


def renumber_files(folder, extension):
    """
    After cleanup, renumber remaining files sequentially based on
    the sorted order of video files in the videos/ folder.
    """
    if not os.path.isdir(folder):
        return

    # Get the canonical video order
    video_files = sorted([
        f for f in os.listdir(video_folder)
        if os.path.isfile(os.path.join(video_folder, f))
        and f.lower().endswith((".mp4", ".webm", ".mkv", ".avi", ".mov"))
    ])

    existing_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(extension)])

    for existing in existing_files:
        norm_existing = normalize_name(existing)
        
        # Find which video this corresponds to
        for idx, video in enumerate(video_files):
            video_norm = os.path.splitext(video)[0].lower().replace(" ", "").replace("-", "").replace("_", "")
            if norm_existing[:30] == video_norm[:30]:
                correct_number = str(idx + 1).zfill(2)
                # Get the name part after the prefix
                parts = os.path.splitext(existing)[0].split("_", 1)
                name_part = parts[1] if len(parts) >= 2 else parts[0]
                correct_name = f"{correct_number}_{name_part}{extension}"

                if existing != correct_name:
                    old_path = os.path.join(folder, existing)
                    new_path = os.path.join(folder, correct_name)
                    
                    # If it's a JSON file, update internal metadata before renaming
                    if extension == ".json":
                        try:
                            with open(old_path, 'r', encoding='utf-8') as f:
                                chunks = json.load(f)
                            for c in chunks:
                                c['number'] = correct_number
                                c['title'] = name_part
                            with open(old_path, 'w', encoding='utf-8') as f:
                                json.dump(chunks, f, indent=4)
                        except Exception as e:
                            print(f"  Warning: Could not update internal JSON {existing}: {e}")

                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"  Renamed: {existing} -> {correct_name}")
                break


def main():
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("=== DRY RUN MODE (no files will be deleted) ===\n")

    print("=" * 50)
    print("  EduQuery — Duplicate Cleanup")
    print("=" * 50)

    # 1. Clean up audio duplicates
    print(f"\n[1/3] Cleaning audio duplicates in '{audio_folder}'...")
    audio_removed = cleanup_folder(audio_folder, ".mp3", dry_run)
    print(f"  -> Removed {audio_removed} duplicate audio file(s)")

    # 2. Clean up JSON duplicates
    print(f"\n[2/3] Cleaning JSON duplicates in '{json_folder}'...")
    json_removed = cleanup_folder(json_folder, ".json", dry_run)
    print(f"  -> Removed {json_removed} duplicate JSON file(s)")

    # 3. Renumber remaining files
    if not dry_run:
        print(f"\n[3/3] Renumbering remaining files...")
        renumber_files(audio_folder, ".mp3")
        renumber_files(json_folder, ".json")

    print(f"\n{'=' * 50}")
    print(f"  Cleanup complete!")
    print(f"  Audio files removed: {audio_removed}")
    print(f"  JSON files removed : {json_removed}")
    print(f"{'=' * 50}")

    # Show final state
    if not dry_run:
        print("\nFinal state:")
        if os.path.isdir(audio_folder):
            audios = sorted(os.listdir(audio_folder))
            print(f"  Audios ({len(audios)}): {audios}")
        if os.path.isdir(json_folder):
            jsons = sorted(os.listdir(json_folder))
            print(f"  JSONs  ({len(jsons)}): {jsons}")


if __name__ == "__main__":
    main()
