"""Tests for the M10 export manifest builder (real artifact wiring)."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qimera.app.champions import ChampionsRegistry
from qimera.app.export_store import (
    export_root, latest_manifest, list_exports, new_manifest,
)


@pytest.fixture
def reg(tmp_path):
    r = ChampionsRegistry(path=tmp_path / "champions.db")
    yield r
    r.close()


def test_manifest_with_empty_registry_uses_skeleton(tmp_path, monkeypatch):
    # Point the registry at an empty temp DB so no champions exist.
    monkeypatch.setattr("qimera.app.export_store.REGISTRY_PATH",
                        tmp_path / "champions.db")
    monkeypatch.setattr("qimera.app.export_store.EXPORT_DIR", tmp_path / "export")

    m = new_manifest("EURUSD", regime="UP_NORMAL", note="test")
    assert m["asset"] == "EURUSD"
    assert m["schema_version"] == 1
    assert m["registry_id"].startswith("chipr-")
    assert len(m["champions"]) == 2  # skeleton fallback
    assert m["champions"][0]["source"] == "Parametric GP"
    # Validation metadata defaults (no log entry for the fresh run id).
    assert m["validation"]["run_id"]
    assert "verdict" not in m["validation"]
    assert m["classifier"]["kind"] == "xgboost"
    assert m["classifier"]["model_path"] is None


def test_manifest_wires_registry_champion(tmp_path, monkeypatch):
    db = tmp_path / "champions.db"
    r = ChampionsRegistry(path=db)
    c = r.promote(asset="EURUSD", regime="UP_NORMAL", source="gp",
                  strategy="cvd", params=[2.0, 20, 1], run_id="run-mc-1",
                  metrics={"net_pnl": 0.5, "win_rate": 0.8, "trades": 12})
    r.close()

    monkeypatch.setattr("qimera.app.export_store.REGISTRY_PATH", db)
    monkeypatch.setattr("qimera.app.export_store.EXPORT_DIR", tmp_path / "export")

    m = new_manifest("EURUSD", regime="UP_NORMAL")
    assert len(m["champions"]) == 1
    champ = m["champions"][0]
    assert champ["strategy"] == "cvd"
    assert champ["params_json"] == [2.0, 20, 1]
    assert champ["metrics_json"]["win_rate"] == 0.8
    assert champ["run_id"] == "run-mc-1"
    # The manifest's registry_id and validation run id follow the champion.
    assert m["registry_id"] == "chipr-run-mc-1"
    assert m["validation"]["run_id"] == "run-mc-1"


def test_manifest_reads_validation_log(tmp_path, monkeypatch):
    # A champion whose run_id has an M4 ValidationLog entry -> real metadata.
    from qimera.app.validation import ValidationLog

    log = ValidationLog(root=tmp_path / "validation")
    log.log({"kind": "walk_forward", "n_folds": 5,
             "verdict": {"passed": True, "reasons": []},
             "aggregate_oos": {"profit_factor": 1.4, "trades": 60}})
    run_id = log.run_id

    db = tmp_path / "champions.db"
    r = ChampionsRegistry(path=db)
    r.promote(asset="XAUUSD", regime="ANY", source="gp", strategy="fvg",
              run_id=run_id)
    r.close()

    monkeypatch.setattr("qimera.app.export_store.REGISTRY_PATH", db)
    monkeypatch.setattr("qimera.app.export_store.EXPORT_DIR", tmp_path / "export")
    # ``_validation_metadata`` reads VALIDATION_ROOT from the validation module
    # at call time, so patch it there.
    monkeypatch.setattr(
        "qimera.app.validation.VALIDATION_ROOT", tmp_path / "validation")

    m = new_manifest("XAUUSD")
    v = m["validation"]
    assert v["run_id"] == run_id
    assert v["kind"] == "walk_forward"
    assert v["verdict"]["passed"] is True
    assert v["aggregate_oos"]["profit_factor"] == 1.4


def test_manifest_reads_classifier_snapshot(tmp_path, monkeypatch):
    snap = tmp_path / "latest_snapshot.json"
    snap.write_text(json.dumps({
        "trained_at": "2026-01-01T00:00:00Z",
        "features": ["returns_1", "regime_code"],
        "feature_importance": {"returns_1": 0.7, "regime_code": 0.3},
        "metrics": {"train_accuracy": 0.58},
    }), encoding="utf-8")

    monkeypatch.setattr("qimera.app.export_store.REGISTRY_PATH",
                        tmp_path / "champions.db")
    monkeypatch.setattr("qimera.app.export_store.EXPORT_DIR", tmp_path / "export")
    monkeypatch.setattr("qimera.app.export_store._CLASSIFIER_SNAPSHOT", snap)

    m = new_manifest("GBPUSD")
    clf = m["classifier"]
    assert clf["kind"] == "xgboost"
    assert clf["feature_importance"]["regime_code"] == 0.3
    assert clf["metrics"]["train_accuracy"] == 0.58


def test_manifest_versioning_and_latest(tmp_path, monkeypatch):
    monkeypatch.setattr("qimera.app.export_store.REGISTRY_PATH",
                        tmp_path / "champions.db")
    monkeypatch.setattr("qimera.app.export_store.EXPORT_DIR", tmp_path / "export")

    m1 = new_manifest("EURUSD")
    m2 = new_manifest("EURUSD")
    assert m2["artifact_id"] != m1["artifact_id"]
    # latest_manifest returns the newest version.
    latest = latest_manifest("EURUSD")
    assert latest["artifact_id"] == m2["artifact_id"]
    # list_exports reports one asset with version 2.
    exports = list_exports()
    eur = next(e for e in exports if e["asset"] == "EURUSD")
    assert eur["version"] == 2


def test_manifest_files_are_written_disk(tmp_path, monkeypatch):
    monkeypatch.setattr("qimera.app.export_store.REGISTRY_PATH",
                        tmp_path / "champions.db")
    monkeypatch.setattr("qimera.app.export_store.EXPORT_DIR", tmp_path / "export")

    m = new_manifest("EURUSD")
    f = tmp_path / "export" / "EURUSD" / "v1" / "manifest.json"
    assert f.is_file()
    on_disk = json.loads(f.read_text(encoding="utf-8"))
    assert on_disk["artifact_id"] == m["artifact_id"]
    assert on_disk["note"] == ""
