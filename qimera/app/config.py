"""QiMera service configuration.

QiMera runs as its own process on its own ports, next to Meruem:
  - API:      127.0.0.1:8766  (this app)
  - Frontend: 127.0.0.1:1421  (browser tab, linked from Meruem's TopNav)
and delivers its export artifacts into Meruem's per-symbol ML-artifact context
store (see the integration contract in ``QiMera.md`` §12).
"""

from pathlib import Path

from pydantic_settings import BaseSettings

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class QiMeraSettings(BaseSettings):
    app_name: str = "QiMera"
    api_port: int = 8766
    frontend_url: str = "http://127.0.0.1:1421"
    # Meruem backend (its /api base) — the injection target for exports.
    meruem_api: str = "http://127.0.0.1:8765/api"
    # Override the export store location; default is <repo>/qimera/data/export.
    export_dir: str = ""

    class Config:
        env_prefix = "QIMERA_"


settings = QiMeraSettings()
EXPORT_DIR = (
    Path(settings.export_dir)
    if settings.export_dir
    else _REPO_ROOT / "qimera" / "data" / "export"
)