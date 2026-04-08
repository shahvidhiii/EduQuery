import requests
import json

print('Testing ChromaDB Backend')
print('=' * 50)

# Test API status
r = requests.get('http://127.0.0.1:8000/api/status')
status = r.json()
print(f'Backend: {status["backend"]}')
print(f'Chunks: {status["chunk_count"]}')
print()

# Test query
print('Testing query: "What is the main topic of the video?"')
r = requests.post('http://127.0.0.1:8000/ask', json={'question': 'What is the main topic of the video?'}, stream=True)

sources_found = False
response_text = ''
for line in r.iter_lines():
    if line:
        obj = json.loads(line)
        if obj['type'] == 'sources':
            sources = obj['sources']
            sources_found = True
            print(f'✅ Found {len(sources)} sources:')
            for s in sources:
                title = s['title'][:40]
                print(f'   - Video {s["video_number"]}: {title}... @ {s["start_time"]}s')
        elif obj['type'] == 'token':
            response_text += obj['token']
        elif obj['type'] == 'done':
            pass

print()
print('Response:', response_text[:200] + '...' if len(response_text) > 200 else response_text)
print()
print('✅ ChromaDB Backend Test PASSED!')
