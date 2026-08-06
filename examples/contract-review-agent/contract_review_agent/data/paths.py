from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
GENERATED_DIR = DATA_DIR / "generated"
CUAD_DIR = GENERATED_DIR / "cuad"
MANIFEST_PATH = GENERATED_DIR / "manifest.json"
DB_PATH = GENERATED_DIR / "obligations.db"
