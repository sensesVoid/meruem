import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_save_reference_creates_reference_doc(client):
    """Saving a concept as a REFERENCE stores a concept-only doc (status reference),
    no QA, and counts the library so synthesis can compile across all references."""
    r = client.post("/api/strategies/reference", json={
        "name": "ICT Playbook",
        "concept_markdown": "Liquidity sweep, then a fair-value gap; CHoCH + FVG retest entry.",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["strategy"]["status"] == "reference"
    assert data["strategy"]["concept_markdown"].startswith("Liquidity sweep")
    assert data["strategy"]["body_markdown"] == ""
    assert data["reference_count"] >= 1

    listed = client.get("/api/strategies").json()
    assert any(s["status"] == "reference" for s in listed)


def test_reference_requires_concept(client):
    r = client.post("/api/strategies/reference", json={"name": "Empty", "concept_markdown": " "})
    assert r.status_code == 422


def test_primitives_list_and_register_validation(client):
    """The registry is listed (GET) and only validated proposals are accepted
    (POST); an invalid proposal is rejected without touching the registry."""
    r = client.get("/api/strategies/primitives")
    assert r.status_code == 200
    names = {p["name"] for p in r.json()["primitives"]}
    assert {"BREAKOUT", "ATR_STOP", "FVG_IMBALANCE"} <= names

    bad = client.post("/api/strategies/primitives/register", json={
        "primitives": [{"name": "not-uppercase", "description": "x",
                         "parameters": {}, "formula": "x"}],
    })
    assert bad.status_code == 400
    assert "name" in (bad.json()["detail"] or "").lower()

    empty = client.post("/api/strategies/primitives/register", json={"primitives": []})
    assert empty.status_code == 400


def test_settings_defaults_created(client):
    r = client.get("/api/settings")
    data = r.json()
    assert data["max_concurrent_auto_trades"] == 2
    assert data["daily_loss_cap_pct"] == -3.0
    assert data["max_drawdown_pct"] == -8.0


def test_update_setting_applies_immediately(client):
    r = client.put("/api/settings", json={"key": "max_concurrent_auto_trades", "value": 5})
    assert r.status_code == 200
    again = client.get("/api/settings").json()
    assert again["max_concurrent_auto_trades"] == 5


def test_account_and_asset_crud(client):
    acc = client.post("/api/accounts", json={
        "id": "acc-master-1", "broker": "MT5", "account_type": "master",
        "nickname": "Prop Master", "risk_scaling_mode": "proportional",
        "scale_factor": 1.0, "encrypted_ref": "mt5:default",
    })
    assert acc.status_code == 200

    slave = client.post("/api/accounts", json={
        "id": "acc-slave-1", "broker": "MT5", "account_type": "slave",
        "nickname": "Slave A", "risk_scaling_mode": "fixed",
        "scale_factor": 0.5, "encrypted_ref": "mt5:slavea",
    })
    assert slave.status_code == 200

    asset = client.post("/api/assets", json={
        "id": "ast-1", "symbol": "XAUUSD", "asset_class": "commodities",
        "timeframe_watchlist": "15m,1h",
    })
    assert asset.status_code == 200

    assert len(client.get("/api/accounts").json()) == 2


def test_vault_roundtrip(client):
    r = client.post("/api/vault", json={
        "namespace": "llm", "username": "gemini", "secret": "test-key-123",
    })
    assert r.status_code == 200
    got = client.get("/api/vault/llm/gemini").json()
    assert got["present"] is True
    assert got["value"] == "test-key-123"


def test_strategy_create_embeds(client):
    r = client.post("/api/strategies", json={
        "name": "VP Breakout v2", "body_markdown": "Buy on volume profile POC breakout with bullish imbalance.",
        "asset_class": "cripto", "timeframe": "15m",
    })
    assert r.status_code == 200


def test_approve_applies_best_refined_params(client):
    strat = client.post("/api/strategies", json={
        "name": "Refined Strat", "body_markdown": "Buy the trend. Stop at ATR.",
        "asset_class": "forex", "timeframe": "15m",
    }).json()
    sid = strat["id"]

    # Simulate a persisted accepted BacktestRun (refine already saved it).
    from sqlmodel import Session

    from app.core.db import engine
    from app.models.entities import BacktestRun

    with Session(engine) as session:
        session.add(BacktestRun(
            id="bt-approve-test", strategy_id=sid, symbol="EURUSD",
            timeframe="15m", iteration=3,
            spec_json='{"fast": 8, "slow": 34, "rsi_period": 11}',
            metrics_json=('{"train": {"n_trades": 40, "win_rate": 55, "sharpe": 1.4, '
                          '"max_drawdown": 8, "profit_factor": 1.3}, '
                          '"test": {"n_trades": 30, "win_rate": 53, "sharpe": 1.2, '
                          '"max_drawdown": 9, "profit_factor": 1.2}}'),
            rationale="best", accepted=1,
        ))
        session.commit()

    r = client.post(f"/api/strategies/{sid}/approve")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "active"
    assert '"fast": 8' in data["params_json"]

    # approve again must not error and keeps the applied spec
    again = client.post(f"/api/strategies/{sid}/approve").json()
    assert again["params_json"] == data["params_json"]


def test_approve_rejects_overfit_metrics(client):
    strat = client.post("/api/strategies", json={
        "name": "Overfit Strat", "body_markdown": "Buy the trend.",
        "asset_class": "forex", "timeframe": "15m",
    }).json()
    sid = strat["id"]

    from sqlmodel import Session

    from app.core.db import engine
    from app.models.entities import BacktestRun

    # High in-sample but OOS not green -> the anti-overfit gate must refuse approval.
    with Session(engine) as session:
        session.add(BacktestRun(
            id="bt-overfit-test", strategy_id=sid, symbol="EURUSD",
            timeframe="15m", iteration=2,
            spec_json='{"fast": 12, "slow": 40, "rsi_period": 14}',
            metrics_json='{"train": {"n_trades": 40, "win_rate": 70, "sharpe": 2.5, '
                          '"max_drawdown": 5, "profit_factor": 2.0}, '
                          '"test": {"n_trades": 30, "win_rate": 45, "sharpe": 0.6, '
                          '"max_drawdown": 20, "profit_factor": 0.9}}',
            rationale="best", accepted=1,
        ))
        session.commit()

    r = client.post(f"/api/strategies/{sid}/approve")
    assert r.status_code == 409
    assert "overfit" in r.json()["detail"].lower()


def test_strategy_submit_starts_job(client):
    r = client.post("/api/strategies/submit", json={
        "name": "Auto Opt",
        "body_markdown": "1. Buy when fast SMA crosses above slow SMA.\n2. Stop below ATR.\n3. Target at ATR multiple.",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["job_id"]

    s = client.get(f"/api/strategies/submit/{body['job_id']}")
    assert s.status_code == 200
    assert s.json()["status"] in ("queued", "running", "done", "error")


def test_submit_requires_concept(client):
    """#8.3: manual entry textbox is retired; a trading concept is now REQUIRED."""
    r = client.post("/api/strategies/submit", json={
        "name": "No Concept",
        "body_markdown": "1. Buy on breakout.",
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    import time

    state = {}
    for _ in range(40):
        state = client.get(f"/api/strategies/submit/{job_id}").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.4)

    assert state["status"] == "error"
    assert "concept" in (state.get("message") or "").lower()


def test_submit_retains_failed_strategy_for_edit(client):
    """A synthesized strategy that FAILs QA is kept (status=failed) so it can be
    edited/discarded."""
    r = client.post("/api/strategies/submit", json={
        "name": "Vague Rule",
        "concept_markdown": "Example only, trust me, insert your own rules here.",
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # Poll until terminal state (QA is heuristic-fast when LLM offline).
    import time

    state = {}
    for _ in range(40):
        state = client.get(f"/api/strategies/submit/{job_id}").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.4)

    assert state["status"] == "error"
    sid = state.get("strategy_id")
    assert sid, "failed pipeline must expose a strategy_id for editing"

    strat = client.get("/api/strategies").json()
    row = next((s for s in strat if s["id"] == sid), None)
    assert row is not None
    assert row["status"] == "failed"
    assert row.get("qa_notes"), "failed strategy should persist QA notes"


def test_submit_ungrounded_synthesis_fails_loudly(client, monkeypatch):
    """An LLM-produced entry that traces to NO ingested source hard-stops the
    pipeline (strategy kept as failed for editing) instead of being backtested
    and silently promoted."""
    from app.agents.synthesizer import SynthesizerAgent

    def fake_synthesize(self, *a, **k):
        return {
            "entry_markdown": "1. Buy the momentum oscillator cross. 2. Stop at risk. 3. Target.",
            "approximations": [],
            "grounded_on": [],
            "summary": "ok",
            "provider": "test",
            "grounded": False,
            "grounding": {
                "score": 0.0,
                "verdict": "UNGROUNDED",
                "ungrounded_rules": ["1. Buy the momentum oscillator cross."],
                "source_count": 1,
            },
            "qa": {"verdict": "PASS", "quality_score": 80, "contamination": False,
                   "issues": [], "provider": "test"},
            "ir": {"status": "SUCCESS", "compiled_blueprint": {"rules": []}},
        }

    monkeypatch.setattr(SynthesizerAgent, "synthesize", fake_synthesize)
    r = client.post("/api/strategies/submit", json={
        "name": "Ungrounded", "concept_markdown": "Momentum on pullbacks.",
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    import time

    state = {}
    for _ in range(40):
        state = client.get(f"/api/strategies/submit/{job_id}").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.4)

    assert state["status"] == "error"
    assert "grounded" in (state.get("message") or "").lower()
    sid = state.get("strategy_id")
    assert sid, "ungrounded failure must still expose a strategy row for editing"
    rows = client.get("/api/strategies").json()
    row = next((s for s in rows if s["id"] == sid), None)
    assert row is not None and row["status"] == "failed"
    assert row.get("qa_notes"), "ungrounded failure should persist QA notes"


def test_submit_backtest_failure_exposes_diagnostics(client, monkeypatch):
    """When synthesis succeeds but backtest/refine fails, the job surfaces the
    actual synthesized strategy, scores, recommendations and thinking trace at
    stage 'optimize' so the UI can render a diagnostics panel."""
    from app.services.backtest_engine import BacktestEngine

    def boom(*a, **k):
        raise ValueError("Only 3 candles available for EURUSD 1h")

    monkeypatch.setattr(BacktestEngine, "refine_code_loop", boom)
    r = client.post("/api/strategies/submit", json={
        "name": "Diag Test",
        "concept_markdown": "Mean reversion on pullbacks to the value area.",
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    import time

    state = {}
    for _ in range(60):
        state = client.get(f"/api/strategies/submit/{job_id}").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.3)

    assert state["status"] == "error"
    assert state.get("stage") == "optimize", (
        "stage must show the backtest failed AFTER a successful synthesis")
    diag = state.get("diagnostics") or {}
    assert diag.get("synth", {}).get("entry_markdown"), (
        "diagnostics must include the synthesized entry criteria")
    assert diag.get("recommendations"), "diagnostics must include recommendations"
    assert diag.get("thinking"), "diagnostics must include the thinking trace"
    steps = {s.get("step") for s in diag["thinking"]}
    assert "retrieval" in steps and "synthesis" in steps


def test_recommendations_derive_from_failure_mode():
    """Recommendations are derived from the actual scores (offline-safe)."""
    from app.services.strategy_pipeline import _recommendations

    recs = _recommendations(gate={
        "n_trades_test": 2, "win_rate_test": 100, "sharpe_test": 3.0,
        "sharpe_train": 0,
    })
    texts = " ".join(r["text"].lower() for r in recs)
    assert "reference" in texts and "entry example" in texts

    weak = _recommendations(gate={
        "n_trades_test": 30, "win_rate_test": 41.0, "sharpe_test": 0.6,
        "sharpe_train": 0.5,
    })
    weak_texts = " ".join(r["text"].lower() for r in weak)
    assert "win rate" in weak_texts or "sharpe" in weak_texts

    overfit = _recommendations(gate={
        "n_trades_test": 30, "win_rate_test": 52.0, "sharpe_test": 0.8,
        "sharpe_train": 2.5,
    })
    assert any(r["type"] == "overfit" for r in overfit)


def test_metrics_green_guard():
    from app.services.strategy_pipeline import _metrics_green, _metrics_overfit_guard

    # Strong OOS -> green even if train is more conservative.
    gate = _metrics_overfit_guard({
        "train": {"n_trades": 40, "win_rate": 40, "sharpe": 0.5,
                  "max_drawdown": 8, "profit_factor": 1.05},
        "test": {"n_trades": 30, "win_rate": 53, "sharpe": 1.4,
                 "max_drawdown": 6, "profit_factor": 1.5},
    })
    assert _metrics_green(gate) is True

    # OOS collapses vs train -> clear overfitting -> reject.
    overfit = _metrics_overfit_guard({
        "train": {"n_trades": 40, "win_rate": 75, "sharpe": 3.1,
                  "max_drawdown": 4, "profit_factor": 2.4},
        "test": {"n_trades": 30, "win_rate": 45, "sharpe": 0.7,
                 "max_drawdown": 18, "profit_factor": 0.8},
    })
    assert _metrics_green(overfit) is False

    # OOS sharpe below 1.0 on its own -> reject regardless of train.
    low_sharpe = _metrics_overfit_guard({
        "train": {"n_trades": 40, "win_rate": 55, "sharpe": 0.9,
                  "max_drawdown": 10, "profit_factor": 1.1},
        "test": {"n_trades": 30, "win_rate": 52, "sharpe": 0.8,
                 "max_drawdown": 11, "profit_factor": 1.0},
    })
    assert _metrics_green(low_sharpe) is False


def test_risk_check(client):
    r = client.get("/api/risk/check")
    assert r.status_code == 200
    assert "allowed" in r.json()


def test_trades_delete_removes_record(client, session):
    """A trade row can be deleted so phantom records don't clutter the UI."""
    from app.models.entities import Account

    session.add(Account(
        id="acct-del", broker="MT5", account_type="slave",
        nickname="del", risk_scaling_mode="proportional", scale_factor=1.0,
        encrypted_ref="", login="0", enabled=True,
    ))
    session.commit()
    r = client.post("/api/trades", json={
        "account_id": "acct-del",
        "direction": "BUY",
        "entry_price": 10.0,
        "sl": 9.9,
        "lot_size": 1.0,
        "mode": "manual",
    })
    trade_id = r.json()["id"]

    d = client.delete(f"/api/trades/{trade_id}")
    assert d.status_code == 200
    assert client.get("/api/trades?open_only=false").json() == []


def test_trades_close_records_reason_and_exit(client, session):
    """Closing a trade stores why it closed + the exit price (LTR context)."""
    from app.models.entities import Account

    session.add(Account(
        id="acct-close", broker="MT5", account_type="slave",
        nickname="close", risk_scaling_mode="proportional", scale_factor=1.0,
        encrypted_ref="", login="0", enabled=True,
    ))
    session.commit()
    r = client.post("/api/trades", json={
        "account_id": "acct-close",
        "direction": "BUY",
        "entry_price": 10.0,
        "sl": 9.8,
        "tp1": 10.5,
        "lot_size": 1.0,
        "mode": "manual",
    })
    trade_id = r.json()["id"]
    c = client.put(f"/api/trades/{trade_id}/close",
                    json={"reason": "sl_hit", "exit_price": 9.9})
    assert c.status_code == 200
    trades = client.get("/api/trades?open_only=false").json()
    closed = next(t for t in trades if t["id"] == trade_id)
    assert closed["close_reason"] == "sl_hit"
    assert closed["exit_price"] == 9.9
    assert closed["closed_at"] is not None


def test_trades_clear_phantoms_only_closed_unconfirmed(client, session):
    """clear-phantoms removes closed rows without a symbol/order id (phantom),
    keeps open and broker-confirmed records."""
    from datetime import datetime, timezone

    from app.models.entities import Account, Trade

    session.add(Account(
        id="acct-ph", broker="MT5", account_type="slave",
        nickname="ph", risk_scaling_mode="proportional", scale_factor=1.0,
        encrypted_ref="", login="0", enabled=True,
    ))
    session.add(Trade(id="real-open", account_id="acct-ph", entry_price=2.0, sl=1.9,
                      direction="BUY", lot_size=0.1, symbol="EURUSD", broker="MT5",
                      closed_at=None))
    session.add(Trade(id="real-confirmed", account_id="acct-ph", entry_price=2.0, sl=1.9,
                      direction="BUY", lot_size=0.1, symbol="EURUSD", broker="MT5",
                      broker_order_id="12345",
                      closed_at=datetime.now(timezone.utc)))
    session.add(Trade(id="phantom-1", account_id="acct-ph", entry_price=1.0, sl=0.9,
                      direction="BUY", lot_size=0.1, symbol="", broker="",
                      closed_at=datetime.now(timezone.utc)))
    session.commit()

    r = client.post("/api/trades/clear-phantoms")
    assert r.status_code == 200
    res = r.json()
    assert res["removed"] == 1
    assert res["ids"] == ["phantom-1"]

    ids = {t["id"] for t in client.get("/api/trades?open_only=false").json()}
    assert "real-open" in ids
    assert "real-confirmed" in ids
    assert "phantom-1" not in ids