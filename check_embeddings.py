import pandas as pd
import os

embeddings_file = 'embeddings.pkl'
if os.path.exists(embeddings_file):
    df = pd.read_pickle(embeddings_file)
    print(f'Total chunks in embeddings.pkl: {len(df)}')
    print(f'\nColumns: {list(df.columns)}')
    print(f'\n--- Unique video numbers ---')
    print(sorted(df['number'].unique()))
    print(f'\n--- Chunks by video number ---')
    print(df['number'].value_counts().sort_index())
    print(f'\n--- Chunks from video 01 ---')
    video_01 = df[df['number'] == 1]
    print(f'✅ Total chunks from video 01: {len(video_01)}')
    if len(video_01) > 0:
        print(f'\nFirst 3 chunks from video 01:')
        for idx, (i, row) in enumerate(video_01.head(3).iterrows()):
            title = row['title'][:50] if pd.notna(row['title']) else 'N/A'
            start = row['start']
            print(f'  {idx+1}. {title}... | Start: {start}s')
else:
    print(f'❌ embeddings.pkl not found')
