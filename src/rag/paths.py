from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "dumps"
INDEX_PATH = PROJECT_ROOT / "storage" / "rag" / "index.sqlite"
