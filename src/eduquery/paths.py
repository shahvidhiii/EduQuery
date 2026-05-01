from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WEB_DIR = PROJECT_ROOT / "web"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"

AUDIOS_DIR = DATA_DIR / "audios"
JSONS_DIR = DATA_DIR / "jsons"
VIDEOS_DIR = DATA_DIR / "videos"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.pkl"
VIDEO_REGISTRY_FILE = DATA_DIR / "video_registry.json"
