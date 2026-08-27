"""Tests for the M9 champions registry (SQLite + rollback utilities)."""

import pytest

from qimera.app.champions import ChampionsRegistry


@pytest.fixture
def reg(tmp_path):
    r = ChampionsRegistry(path=tmp_path / "champions.db")
    yield r
    r.close()


def _champion(asset="EURUSD", regime="UP_NORMAL", source="gp",
              strategy="cvd", params=None, run_id="run123", metrics=None,
              manifest_version=1):
    return dict(
        asset=asset, regime=regime, source=source, strategy=strategy,
        params=params if params is not None else [2.0, 20, 1],
        run_id=run_id,
        metrics=metrics if metrics is not None else {"net_pnl": 0.5, "trades": 12},
        manifest_version=manifest_version)


def test_promote_and_get(reg):
    c = reg.promote(**_champion())
    assert c["id"] > 0
    assert c["promotion"] == "promoted"
    assert c["asset"] == "EURUSD"
    assert c["params_json"] == [2.0, 20, 1]
    assert c["metrics_json"]["net_pnl"] == 0.5
    got = reg.get(c["id"])
    assert got["strategy"] == "cvd"


def test_promote_supersedes_previous(reg):
    c1 = reg.promote(**_champion(run_id="r1"))
    c2 = reg.promote(**_champion(run_id="r2"))
    assert c2["superseded_id"] == c1["id"]
    assert reg.get(c1["id"])["promotion"] == "superseded"
    assert reg.get(c2["id"])["promotion"] == "promoted"
    assert reg.current("EURUSD", "UP_NORMAL")["id"] == c2["id"]


def test_promote_is_scoped_to_asset_and_regime(reg):
    c1 = reg.promote(**_champion(asset="EURUSD", regime="UP_NORMAL"))
    c2 = reg.promote(**_champion(asset="EURUSD", regime="DOWN_NORMAL"))
    # Different regime -> no supersede.
    assert c2["superseded_id"] is None
    assert reg.current("EURUSD", "UP_NORMAL")["id"] == c1["id"]
    assert reg.current("EURUSD", "DOWN_NORMAL")["id"] == c2["id"]
    # Different asset too.
    c3 = reg.promote(**_champion(asset="XAUUSD", regime="UP_NORMAL"))
    assert c3["superseded_id"] is None


def test_lifecycle_supersede_and_reject(reg):
    c = reg.promote(**_champion())
    assert reg.supersede(c["id"])["promotion"] == "superseded"
    c2 = reg.promote(**_champion())
    assert reg.reject(c2["id"], note="bad PF")["promotion"] == "rejected"
    assert reg.get(c2["id"])["metrics_json"].get("reject_note") == "bad PF"


def test_rollback_restores_latest_superseded(reg):
    c1 = reg.promote(**_champion(run_id="r1"))
    reg.promote(**_champion(run_id="r2"))  # supersedes c1
    restored = reg.rollback("EURUSD", "UP_NORMAL")
    assert restored is not None
    assert restored["id"] == c1["id"]
    assert restored["promotion"] == "promoted"
    assert reg.current("EURUSD", "UP_NORMAL")["id"] == c1["id"]
    # The formerly-promoted c2 is now superseded.
    assert reg.list_champions(promotion="superseded")[0]["run_id"] == "r2"


def test_rollback_to_specific_target(reg):
    c1 = reg.promote(**_champion(run_id="r1"))
    reg.promote(**_champion(run_id="r2"))
    c3 = reg.promote(**_champion(run_id="r3"))
    restored = reg.rollback("EURUSD", "UP_NORMAL", target_id=c1["id"])
    assert restored["id"] == c1["id"]
    assert reg.get(c3["id"])["promotion"] == "superseded"


def test_rollback_target_mismatch_raises(reg):
    reg.promote(**_champion(asset="EURUSD"))
    other = reg.promote(**_champion(asset="XAUUSD"))
    with pytest.raises(ValueError):
        reg.rollback("EURUSD", "UP_NORMAL", target_id=other["id"])


def test_rollback_with_no_history_returns_none(reg):
    assert reg.rollback("EURUSD", "UP_NORMAL") is None


def test_list_and_history(reg):
    reg.promote(**_champion(run_id="r1"))
    reg.promote(**_champion(run_id="r2"))
    reg.promote(**_champion(asset="XAUUSD", run_id="r3"))
    assert len(reg.list_champions()) == 3
    assert len(reg.list_champions(asset="EURUSD")) == 2
    assert len(reg.list_champions(promotion="promoted")) == 2
    hist = reg.history("EURUSD", "UP_NORMAL")
    assert [h["run_id"] for h in hist] == ["r2", "r1"]  # newest first


def test_metrics_json_round_trip(reg):
    c = reg.promote(**_champion(metrics={"net_pnl": 1.5, "trades": 20}))
    assert c["metrics_json"] == {"net_pnl": 1.5, "trades": 20}
