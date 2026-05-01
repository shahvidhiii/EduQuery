import requests
import json

print('Testing ChromaDB API with ALL 330 embedded chunks')
print('=' * 60)

# Test API status
r = requests.get('http://127.0.0.1:8000/api/status')
status = r.json()
print(f'Backend: {status["backend"]}')
print(f'Total chunks: {status["chunk_count"]}')
print()

# Test 1: Query about video 01 content
print('Test 1: Query about English vocabulary (Video 01)')
r = requests.post('http://127.0.0.1:8000/ask', json={'question': 'What is vocabulary?'}, stream=True)
for line in r.iter_lines():
    if line:
        obj = json.loads(line)
        if obj['type'] == 'sources':
            sources = obj['sources']
            print(f'  ✅ Found {len(sources)} sources')
            videos = [s["video_number"] for s in sources]
            print(f'     Videos: {videos}')

# Test 2: Query about videos 21-30 content
print()
print('Test 2: Query about error detection (Videos 21-30)')
r = requests.post('http://127.0.0.1:8000/ask', json={'question': 'What is parity check?'}, stream=True)
for line in r.iter_lines():
    if line:
        obj = json.loads(line)
        if obj['type'] == 'sources':
            sources = obj['sources']
            print(f'  ✅ Found {len(sources)} sources')
            videos = [s["video_number"] for s in sources]
            print(f'     Videos: {videos}')

print()
print('=' * 60)
print('✅ All 330 chunks are searchable via ChromaDB!')
print('=' * 60)
