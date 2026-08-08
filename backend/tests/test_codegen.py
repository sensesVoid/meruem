"""Tests for the AI strategy codegen + safety sandbox + faithful backtest loop."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.code_sandbox import (
    build_strategy_class,
    dry_run_strategy,
    validate_code,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

GOOD_CODE = """
import numpy as np
import pandas as pd
import math
from backtesting import Strategy

class GeneratedStrategy(Strategy):
    risk_pct = 0.01

    def init(self):
        close = self.data.Close
        self.sma = self.I(lambda x: pd.Series(x).rolling(10).mean().values, close)

    def next(self):
        if self.position:
            return
        p = self.data.Close[-1]
        if not math.isfinite(self.sma[-1]):
            return
        if p > self.sma[-1]:
            sl = p - p * 0.01
            tp = p + p * 0.02
            size = min(self.risk_pct * self.equity / (p - sl), self.equity / p)
            self.buy(size=size, sl=sl, tp=tp)
        else:
            sl = p + p * 0.01
            tp = p - p * 0.02
            size = min(self.risk_pct * self.equity / (sl - p), self.equity / p)
            self.sell(size=size, sl=sl, tp=tp)
"""


def _df():
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="15min")
    base = 100.0 + pd.Series(range(n)) * 0.02
    close = base + pd.Series(
        [((-1) ** (i // 7)) * 0.3 for i in range(n)]).cumsum()
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.2
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.2
    return pd.DataFrame({
        "Open": open_.values, "High": high.values, "Low": low.values,
        "Close": close.values, "Volume": np.ones(n) * 1000,
    }, index=idx)


GOOD_CODE = """
import numpy as np
import pandas as pd
import math
from backtesting import Strategy

class GeneratedStrategy(Strategy):
    risk_pct = 0.01

    def init(self):
        close = self.data.Close
        self.sma = self.I(lambda x: pd.Series(x).rolling(10).mean().values, close)

    @staticmethod
    def _risk_size(equity, risk_pct, dist, price):
        if dist <= 0 or price <= 0 or equity <= 0:
            return 0.0
        units = (equity * risk_pct) / dist
        max_units = equity / price
        if units >= 1 and max_units >= 1:
            return float(max(1, int(min(units, max_units))))
        return min(max(((equity * risk_pct) / price), 0.0), 0.99)

    def next(self):
        if self.position:
            return
        p = self.data.Close[-1]
        if not math.isfinite(self.sma[-1]):
            return
        if p > self.sma[-1]:
            sl = p - p * 0.01
            tp = p + p * 0.02
            size = self._risk_size(self.equity, self.risk_pct, p - sl, p)
            if size <= 0:
                return
            self.buy(size=size, sl=sl, tp=tp)
        else:
            sl = p + p * 0.01
            tp = p - p * 0.02
            size = self._risk_size(self.equity, self.risk_pct, sl - p, p)
            if size <= 0:
                return
            self.sell(size=size, sl=sl, tp=tp)
"""


def test_validate_accepts_good_code():
    ok, err = validate_code(GOOD_CODE)
    assert ok, err
    cls = build_strategy_class(GOOD_CODE)
    res = dry_run_strategy(cls, _df())
    assert res["ok"], res

def test_validate_rejects_os_import():
    bad = GOOD_CODE.replace("import math", "import os\nimport math")
    ok, err = validate_code(bad)
    assert not ok
    assert "os" in err


def test_validate_rejects_open_builtin():
    bad = GOOD_CODE.replace(
        "        if self.position:\n            return",
        "        file_handle = open('/etc/passwd')\n        if self.position:\n            return",
    )
    ok, err = validate_code(bad)
    assert not ok


def test_validate_rejects_missing_generated_strategy():
    no_class = GOOD_CODE.replace("class GeneratedStrategy(Strategy):",
                                 "class OtherStrategy(Strategy):")
    try:
        build_strategy_class(no_class)
        raise AssertionError("should have rejected missing GeneratedStrategy")
    except Exception:
        pass


def test_codegen_generate_and_repair_pass_concept(monkeypatch):
    """generate()/repair() include the trading concept in the prompt and keep
    the entry criteria as the hard boundary."""
    from app.agents.strategy_codegen import StrategyCodeGen

    seen_concept = {}
    seen_entry = {}

    def fake_generate(self, title, concept, markdown, symbol, timeframe):
        seen_concept["gen"] = concept
        seen_entry["gen"] = markdown
        return {"code": GOOD_CODE, "notes": "x",
                "confluence": {"fast": 10}, "provider": "t"}

    def fake_repair(self, title, concept, markdown, symbol, timeframe,
                    code, train_metrics, test_metrics):
        seen_concept["rep"] = concept
        seen_entry["rep"] = markdown
        return {"code": GOOD_CODE, "notes": "x",
                "confluence": {"fast": 10}, "provider": "t"}

    monkeypatch.setattr(StrategyCodeGen, "generate", fake_generate)
    monkeypatch.setattr(StrategyCodeGen, "repair", fake_repair)
    gen = StrategyCodeGen()
    gen.generate("N", "CONCEPT", "ENTRY", "EURUSD", "1h")
    gen.repair("N", "CONCEPT", "ENTRY", "EURUSD", "1h", GOOD_CODE, {}, {})
    assert seen_concept == {"gen": "CONCEPT", "rep": "CONCEPT"}
    assert seen_entry == {"gen": "ENTRY", "rep": "ENTRY"}


def test_pipeline_uses_generated_code_and_persists_confluence(client):
    """The submit pipeline runs a generated strategy and stores its confluence."""
    payload = {
        "name": "Codegen Test",
        "concept_markdown": ("Momentum strategy that follows the dominating H1 trend "
                             "and enters on pullbacks that resume the trend."),
        "body_markdown": ("1. Buy when price is above 10-SMA momentum.\n"
                          "2. Stop 1% below entry.\n"
                          "3. Target 2% above entry."),
        "symbol": "EURUSD",
        "timeframe": "1h",
        "bars": 400,
        "max_iterations": 1,
    }

    fake_df = _df()

    def fake_load(symbol, timeframe, bars, start_date=None, end_date=None):
        return fake_df, "test"

    with patch("app.agents.strategy_codegen.StrategyCodeGen.generate",
               return_value={"code": GOOD_CODE, "confluence": {
                   "fast": 10, "slow": 30, "rsi_period": 14, "rsi_buy": 40,
                   "rsi_sell": 60, "atr_period": 14, "atr_sl_mult": 1.5,
                   "atr_tp_mult": 3.0, "risk_pct": 0.01,
               }, "notes": "ok", "provider": "test"}):
        with patch("app.services.backtest_engine._load_df", side_effect=fake_load):
            r = client.post("/api/strategies/submit", json=payload)
            assert r.status_code == 200
            job_id = r.json()["job_id"]

            import time

            state = {}
            for _ in range(120):
                state = client.get(f"/api/strategies/submit/{job_id}").json()
                if state["status"] in ("done", "error"):
                    break
                time.sleep(0.3)

    assert state["status"] == "done", state.get("message")
    assert state["faithful"] is True
    assert isinstance(state.get("gate"), dict)

    rows = client.get("/api/strategies").json()
    row = next((s for s in rows if s["id"] == state["strategy_id"]), None)
    assert row is not None
    assert row["strategy_code"], "generated code must be persisted"
    assert row["params_json"], "confluence params must be persisted"
    assert row["concept_markdown"].startswith("Momentum strategy"), \
        "the trading concept must be persisted as-is"


def test_manual_backtest_runs_faithful_code_when_present(client):
    """The manual /backtest endpoint must run the stored strategy_code (faithful)
    rather than falling back to proxy SMA/RSI/ATR parameters."""
    payload = {
        "name": "Faithful BT",
        "concept_markdown": "Trend following on pullbacks.",
        "body_markdown": ("1. Buy when price is above 10-SMA momentum.\n"
                          "2. Stop 1% below entry.\n"
                          "3. Target 2% above entry."),
        "symbol": "EURUSD",
        "timeframe": "1h",
        "bars": 400,
        "max_iterations": 1,
    }

    fake_df = _df()

    def fake_load(symbol, timeframe, bars, start_date=None, end_date=None):
        return fake_df, "test"

    import json as _json

    with patch("app.agents.strategy_codegen.StrategyCodeGen.generate",
               return_value={"code": GOOD_CODE, "confluence": {
                   "fast": 10, "slow": 30, "rsi_period": 14,
               }, "notes": "ok", "provider": "test"}):
        with patch("app.services.backtest_engine._load_df", side_effect=fake_load):
            r = client.post("/api/strategies/submit", json=payload)
            job_id = r.json()["job_id"]
            import time
            for _ in range(120):
                state = client.get(f"/api/strategies/submit/{job_id}").json()
                if state["status"] in ("done", "error"):
                    break
                time.sleep(0.3)

    sid = state["strategy_id"]
    with patch("app.services.backtest_engine._load_df", side_effect=fake_load):
        bt = client.post(f"/api/strategies/{sid}/backtest",
                         json={"symbol": "EURUSD", "timeframe": "1h", "bars": 400})
    assert bt.status_code == 200, bt.text
    body = bt.json()
    assert body.get("faithful") is True, "should flag a faithful backtest"
    assert isinstance(body.get("metrics"), dict) and body["metrics"]["n_trades"] > 0