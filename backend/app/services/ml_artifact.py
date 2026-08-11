"""ML artifact context-file service.

Traders can upload model/ML context files (model notes, backtest dumps, LLM
summaries, .txt/.md/.json/.csv) that the scanner consumes as *context only* at
scan time. Artifacts live on disk under ``backend/data/ml_artifacts`` and may be
optionally scoped to a symbol so scans of that symbol only see its own files.
"""

import json
import os
import re
import uuid
from pathlib import Path

_ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ml_artifacts"
_MAX_BYTES = 512 * 1024  # per file
_ALLOWED_EXT = {".txt", ".md", ".json", ".csv"}
_MAX_PREVIEW_CHARS = 6000  # per file injected into the scanner prompt


def _ensure_dir() -> Path:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return _ARTIFACT_DIR


def _clean_symbol(symbol: str | None) -> str:
    """Normalize an optional symbol scope to a safe folder name (uppercased)."""
    if not symbol:
        return ""
    safe = re.sub(r"[^A-Z0-9]", "", str(symbol).upper())
    return safe


def save(filename: str, content: bytes, symbol: str | None = None) -> dict:
    """Persist one ML context file. Raises ValueError on invalid input."""
    if not filename:
        raise ValueError("filename is required")
    name = Path(filename).name  # strip any path traversal
    ext = Path(name).suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise ValueError(f"unsupported file type {ext or '(none)'}; "
                         f"allowed: {', '.join(sorted(_ALLOWED_EXT))}")
    if not content:
        raise ValueError("file is empty")
    if len(content) > _MAX_BYTES:
        raise ValueError(f"file too large (max {_MAX_BYTES} bytes)")
    symbol = _clean_symbol(symbol)
    base = _ensure_dir()
    folder = base / symbol if symbol else base
    folder.mkdir(parents=True, exist_ok=True)
    art_id = uuid.uuid4().hex
    stored = folder / f"{art_id}_{name}"
    stored.write_bytes(content)
    return {
        "id": art_id,
        "filename": name,
        "symbol": symbol or None,
        "size": len(content),
    }


def list_artifacts(symbol: str | None = None) -> list[dict]:
    """All artifacts, newest first; optionally filtered to one symbol scope."""
    base = _ensure_dir()
    out: list[dict] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        symbol_folder = rel.parts[0] if len(rel.parts) > 1 else ""
        if symbol is not None and _clean_symbol(symbol) != symbol_folder:
            continue
        art_id = p.stem.split("_", 1)[0]
        original = p.name[len(art_id) + 1:]
        out.append({
            "id": art_id,
            "filename": original,
            "symbol": symbol_folder or None,
            "size": p.stat().st_size,
            "created_at": p.stat().st_mtime,
        })
    out.sort(key=lambda a: a["created_at"], reverse=True)
    return out


def contents(artifact_id: str) -> dict | None:
    """Preview text for one artifact (capped), for scanner context + UI preview."""
    base = _ensure_dir()
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if not p.name.startswith(artifact_id + "_"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - binary-ish files degrade to preview
            text = "<binary / unreadable content>"
        rel = p.relative_to(base)
        symbol_folder = rel.parts[0] if len(rel.parts) > 1 else ""
        return {
            "id": artifact_id,
            "filename": p.name[len(artifact_id) + 1:],
            "symbol": symbol_folder or None,
            "size": p.stat().st_size,
            "preview": text[:_MAX_PREVIEW_CHARS],
        }
    return None


def delete(artifact_id: str) -> bool:
    base = _ensure_dir()
    for p in base.rglob("*"):
        if p.is_file() and p.name.startswith(artifact_id + "_"):
            p.unlink(missing_ok=True)
            return True
    return False


def context_for(symbol: str | None) -> list[dict]:
    """Artifacts to hand the scanner for a symbol: symbol-scoped files first,
    then any global files. Each entry is a compact {filename, content} pair."""
    if not symbol:
        return []
    scoped = [a for a in list_artifacts(symbol=symbol)]
    global_arts = [a for a in list_artifacts() if not a["symbol"]]
    out: list[dict] = []
    for art in scoped + global_arts:
        c = contents(art["id"])
        if c and c.get("preview"):
            out.append({
                "filename": c["filename"],
                "symbol": c["symbol"],
                "content": c["preview"],
            })
    return out


def _serialize_summary(artifacts: list[dict]) -> str:
    return json.dumps(artifacts, ensure_ascii=False)[:_MAX_PREVIEW_CHARS]
