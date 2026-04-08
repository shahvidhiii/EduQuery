import requests
import json

print("=" * 70)
print("  Testing API /ask endpoint for 'What is error correction?'")
print("=" * 70)
print()

# Query the API
r = requests.post(
    'http://127.0.0.1:8000/ask',
    json={'question': 'What is error correction?'},
    stream=True
)

print("API Response (first 10 lines):")
print()

count = 0
for line in r.iter_lines():
    if line:
        obj = json.loads(line)
        
        if obj['type'] == 'sources':
            sources = obj['sources']
            print(f"SOURCES RETURNED BY API ({len(sources)} total):")
            for i, s in enumerate(sources):
                print(f"  {i+1}. Video {s['video_number']}: {s['title'][:40]}... @ {s['start_time']}s")
            print()
        
        elif obj['type'] == 'token':
            if count < 30:  # First 30 tokens
                print(obj['token'], end='', flush=True)
            count += 1
        
        elif obj['type'] == 'done':
            print()
            print()
            print("=" * 70)

print("=" * 70)
print("Check: Are the sources from Videos 27-30 or Video 01?")
print("=" * 70)
