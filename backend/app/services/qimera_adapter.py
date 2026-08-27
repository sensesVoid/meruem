"""QiMera adapter — reads QiMera export artifacts into Meruem's scanner context.

QiMera (the ``qimera/`` subproject) is an offline research/optimization terminal
that exports a versioned, self-describing artifact per asset. This adapter
surfaces the latest valid manifest for a scanned symbol as *context only* to the
scanner — same contract as the ML-artifact files, so the champion feature params
from QiMera sharpen Meruem's reads without ever overriding approved rules.

Everything is read-only and best-effort: if QiMera's export store is absent or a
manifest is missing/unsupported, the scan proceeds exactly as before.
"""

import json
from pathlib import Path

from app.core.config import settings

# Keep the injected context bounded — the scanner prompt is already tight.
_MAX_CONTEXT_CHARS = 6000


def _repo_root() -> Path:
    # backend/app/services/qimera_adapter.py -> parents[3] = repo root
    return Path(__file__).resolve().parent.parent.parent.parent


def export_root() -> Path:
    configured = (settings.qimera_export_dir or "").strip()
    if configured:
        return Path(configured)
    return _repo_root() / "qimera" / "data" / "export"


def latest_manifest(asset: str) -> dict | None:
    """Newest schema-valid manifest for an asset, or None."""
    asset_dir = export_root() / asset.upper().replace(" ", "-")
    if not asset_dir.is_dir():
        return None
    versions = sorted(
        (d for d in asset_dir.glob("v*") if d.is_dir() and d.name[1:].isdigit()),
        key=lambda d: int(d.name[1:]),
        reverse=True,
    )
    for version in versions:
        f = version / "manifest.json"
        if not f.is_file():
            continue
        try:
            manifest = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if int(manifest.get("schema_version") or 0) >= 1:
            return manifest
    return None


def context_for(symbol: str | None) -> list[dict]:
    """Manifests to hand the scanner for ``symbol``: [{filename, content}]."""
    if not symbol:
        return []
    try:
        manifest = latest_manifest(symbol)
    except Exception:  # noqa: BLE001 - scan must never depend on QiMera
        return []
    if not manifest:
        return []
    version = manifest.get("registry_id") or manifest.get("artifact_id", "")[:10]
    content = json.dumps(manifest, ensure_ascii=False)[:_MAX_CONTEXT_CHARS]
    return [{
        "filename": f"qimera_{symbol.upper()}_{version}.json",
        "symbol": symbol.upper(),
        "source": "qimera",
        "content": content,
    }]


def status() -> dict:
    """Adapter status for the UI (QiMera link button + injection diagnostics)."""
    root = export_root()
    assets: list[str] = []
    try:
        if root.is_dir():
            assets = sorted(
                d.name for d in root.iterdir() if d.is_dir() and latest_manifest(d.name)
            )
    except Exception:  # noqa: BLE001, S110 - best-effort scan of the export dir
        pass
    reachable = False
    try:
        import requests

        resp = requests.get(f"{settings.qimera_api_url}/health", timeout=1.5)
        reachable = resp.ok
    except Exception:  # noqa: BLE001 - best-effort connectivity probe
        reachable = False
    return {
        "frontend_url": settings.qimera_url,
        "api_url": settings.qimera_api_url,
        "reachable": reachable,
        "export_dir": str(root),
        "export_dir_exists": root.is_dir(),
        "assets": assets,
    }