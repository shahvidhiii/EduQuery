# EduQuery Folder Structure

EduQuery is organized by responsibility instead of keeping source, scripts, UI, tests, and generated data in the repository root.

```text
src/eduquery/      Backend application package
scripts/           Pipeline, migration, diagnostics, and verification scripts
tests/             API and ChromaDB smoke tests
web/               Browser-facing HTML files
docs/              Project notes, prompts, and migration documentation
data/              Local runtime data and generated artifacts
```

The `data/` folder is intentionally gitignored because it contains generated chunks, vector indexes, local media, and large pickle files.

Root-level launchers are kept for the existing workflow:

```powershell
python main.py
python process_video.py
python create_chunks.py
python read_chunks.py
python read_chunks_v2.py
```

Internally, those wrappers delegate to `src/eduquery/app.py` or `scripts/`.
