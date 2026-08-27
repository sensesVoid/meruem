import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path  # noqa: E402

from pydantic_settings import BaseSettings  # noqa: E402

# Root of the backend package. All relative file paths (DB, chroma) are
# anchored here so the app behaves identically regardless of the current
# working directory (e.g. launching from the repo root vs from backend/),
# preventing data from silently splitting across multiple SQLite files.
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "Meruem"
    host: str = "127.0.0.1"
    port: int = 8765
    database_url: str = f"sqlite:///{BACKEND_DIR / 'meruem.db'}"
    chroma_dir: str = str(BACKEND_DIR / "chroma_data")
    embeddings_model: str = "all-MiniLM-L6-v2"

    gemini_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_scanner_model: str = "qwen3:4b"
    # Auto-detect the Ollama endpoint + installed models at launch (best-effort,
    # non-blocking). Off in tests so the suite never probes the network.
    ollama_detect_on_launch: bool = True

    scan_interval_seconds: int = 300
    data_retention_days: int = 90

    # LTR automation: auto-review once this many losing trades are pending.
    ltr_review_threshold: int = 10
    ltr_check_interval_seconds: int = 3600

    # QiMera extension (research/optimization subproject, runs separately).
    # https://127.0.0.1:1421 is its web UI; the export store's directory feeds
    # the QiMera manifests into the scanner's per-symbol ML-artifact context.
    qimera_url: str = "http://127.0.0.1:1421"
    qimera_api_url: str = "http://127.0.0.1:8766/api"
    qimera_export_dir: str = ""  # empty = <repo>/qimera/data/export

    class Config:
        env_file = str(BACKEND_DIR / ".env")
        env_prefix = "MERUEM_"


settings = Settings()