from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).parent / "scripts" / "migrate_to_chromadb.py"), run_name="__main__")
