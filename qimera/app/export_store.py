"""Export artifact store (QiMera.md §6.12).

QiMera's output is a versioned, self-describing artifact package per asset.
The manifest builder (``new_manifest``) now wires in the **real** research
artifacts instead of the M10-skeleton champions:

- champions: the currently promoted M9 ``ChampionsRegistry`` entries for the
  asset (falls back to the historical skeleton when the registry is empty, so
  the Meruem injection seam keeps working out of the box).
- validation: real walk-forward + Monte-Carlo metadata when the champion's
  ``run_id`` has an M4 ``ValidationLog`` entry, else the schema defaults.
- classifier: real M8 model metadata (kind, trained_at, feature importance)
  when a saved classifier snapshot exists.

Layout::

    qimera/data/export/
      EURUSD/
        v1/manifest.json
        v2/manifest.json
        ...
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qimera.app.champions import REGISTRY_PATH, ChampionsRegistry
from qimera.app.config import EXPORT_DIR

MANIFEST_SCHEMA = 1

_CLASSIFIER_SNAPSHOT = (
    Path(__file__).resolve().parent.parent / "data" / "classifier"
    / "latest_snapshot.json"
)


def export_root() -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORT_DIR


def _sample_champions(asset: str, regime: str | None) -> list[dict]:
    """Skeleton champions — only used when the M9 registry is empty."""
    return [
        {
            "candidate_id": uuid.uuid4().hex[:12],
            "feature": "vwap_band",
            "params": {"width_pct": 0.18, "anchor": "session", "recalc": "bar"},
            "regime": regime,
            "source": "Parametric GP",
        },
        {
            "candidate_id": uuid.uuid4().hex[:12],
            "feature": "fvg_min_gap",
            "params": {"gap_pips": 12.0, "direction": "neutral", "fill_rule": "full"},
            "regime": regime,
            "source": "Indicator Interpretation",
        },
    ]


def _registry_champions(asset: str, regime: str | None) -> list[dict]:
    """Currently promoted M9 champions for the asset (all regimes)."""
    reg = ChampionsRegistry(path=REGISTRY_PATH)
    try:
        return reg.list_champions(asset=asset, promotion="promoted")
    finally:
        reg.close()


def _validation_metadata(run_id: str) -> dict:
    """Real M4 walk-forward/MC metadata for ``run_id`` when a log entry exists."""
    from qimera.app.validation import VALIDATION_ROOT

    path = VALIDATION_ROOT / f"{run_id}.jsonl"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return {}
    if not entries:
        return {}
    last = entries[-1]
    return {
        "run_id": run_id,
        "kind": last.get("kind"),
        "verdict": last.get("verdict"),
        "aggregate_oos": last.get("aggregate_oos"),
        "n_folds": last.get("n_folds"),
    }


def _classifier_metadata() -> dict:
    """Real M8 classifier snapshot metadata when one has been saved."""
    if not _CLASSIFIER_SNAPSHOT.is_file():
        return {"kind": "xgboost", "model_path": None}
    try:
        snap = json.loads(_CLASSIFIER_SNAPSHOT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"kind": "xgboost", "model_path": None}
    return {
        "kind": "xgboost",
        "model_path": str(_CLASSIFIER_SNAPSHOT),
        "trained_at": snap.get("trained_at"),
        "features": snap.get("features"),
        "feature_importance": snap.get("feature_importance"),
        "metrics": snap.get("metrics"),
    }


def new_manifest(asset: str, regime: str | None = None, note: str = "") -> dict:
    """Create a versioned manifest file for ``asset`` and return it.

    Wires the real M9 champions + M4 validation + M8 classifier metadata into
    the manifest; falls back to the skeleton when those stores are empty.
    """
    asset = asset.upper().replace(" ", "-")
    asset_dir = export_root() / asset
    existing = sorted(
        (d for d in asset_dir.glob("v*") if d.is_dir()),
        key=lambda d: int(d.name[1:]),
    )
    version = (int(existing[-1].name[1:]) + 1) if existing else 1
    run_id = uuid.uuid4().hex[:16]

    champions = _registry_champions(asset, regime) or _sample_champions(asset, regime)
    first_champion = champions[0] if champions else {}
    champion_run_id = first_champion.get("run_id") or run_id

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "artifact_id": uuid.uuid4().hex,
        "asset": asset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "qimera",
        "regime": regime,
        "champions": champions,
        "feature_library_version": "0.0.1",
        "time_decay": {"lambda": 0.02, "window_days": 90},
        "validation": {
            "run_id": champion_run_id,
            "walk_forward": True,
            "mc_iterations": 200,
            "date_range": [None, None],
            "thresholds": {"min_win_rate": 0.52, "max_drawdown_pct": -8.0},
            **{k: v for k, v in _validation_metadata(champion_run_id).items()
               if k not in ("run_id",)},
        },
        "classifier": _classifier_metadata(),
        "registry_id": f"chipr-{champion_run_id}",
        "note": note,
    }
    v_dir = asset_dir / f"v{version}"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def list_exports() -> list[dict]:
    """Per-asset summaries: latest version, generated_at, champion count."""
    root = export_root()
    out: list[dict] = []
    for asset_dir in sorted(root.iterdir()):
        if not asset_dir.is_dir():
            continue
        m = latest_manifest(asset_dir.name)
        if m is None:
            continue
        out.append({
            "asset": m["asset"],
            "version": _latest_version(asset_dir),
            "generated_at": m.get("generated_at"),
            "champions": len(m.get("champions") or []),
            "artifact_id": m.get("artifact_id"),
        })
    out.sort(key=lambda x: x["asset"])
    return out


def _latest_version(asset_dir: Path) -> int | None:
    versions = [
        d for d in asset_dir.glob("v*") if d.is_dir() and d.name[1:].isdigit()
    ]
    if not versions:
        return None
    return max(int(d.name[1:]) for d in versions)


def latest_manifest(asset: str) -> dict | None:
    """Newest valid manifest for an asset, or None (missing / unsupported schema)."""
    asset_dir = export_root() / asset.upper().replace(" ", "-")
    if not asset_dir.is_dir():
        return None
    for version in sorted(
        (d for d in asset_dir.glob("v*") if d.is_dir() and d.name[1:].isdigit()),
        key=lambda d: int(d.name[1:]),
        reverse=True,
    ):
        f = version / "manifest.json"
        if not f.is_file():
            continue
        try:
            manifest = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if int(manifest.get("schema_version") or 0) >= MANIFEST_SCHEMA:
            return manifest
    return None