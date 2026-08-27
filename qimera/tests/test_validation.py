"""Tests for the M4 walk-forward validation layer and Monte Carlo gate."""

import numpy as np
import pandas as pd
import pytest

from qimera.app import julia_bridge as jb
from qimera.app.validation import (
    ValidationLog, bars_per_day, mc_max_drawdowns, monte_carlo_gate,
    oos_degradation_check, run_walk_forward, walk_forward_folds,
)


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------


def test_bars_per_day():
    assert bars_per_day("15m") == 96
    assert bars_per_day("1m") == 1440
    assert bars_per_day("1h") == 24
    assert bars_per_day("4h") == 6


def test_walk_forward_folds_layout():
    n = 1000
    folds = walk_forward_folds(n, n_folds=5, train_frac=0.6, purge_bars=96)
    assert len(folds) == 5

    for i, f in enumerate(folds):
        t_start, t_end = f["train"]
        o_start, o_end = f["test"]
        assert t_start == 0                                  # expanding train
        assert t_end < o_start                               # gap separates
        assert o_start - t_end == 96                         # 1-day purged gap
        assert o_end <= n
        assert o_end - o_start >= 20
        assert t_end >= 100

    # strictly chronological OOS blocks, no overlap
    ends = [f["test"][1] for f in folds]
    starts = [f["test"][0] for f in folds]
    assert ends == sorted(ends)
    assert all(starts[i + 1] == ends[i] for i in range(len(starts) - 1))


def test_walk_forward_folds_too_short_raises():
    with pytest.raises(ValueError):
        walk_forward_folds(n=150, n_folds=5, train_frac=0.5, min_train=100)
    with pytest.raises(ValueError):
        walk_forward_folds(n=1000, n_folds=1)


# ---------------------------------------------------------------------------
# OOS degradation check
# ---------------------------------------------------------------------------


def _metrics(pf, wr, dd):
    return {"profit_factor": pf, "win_rate": wr, "max_dd_pct": dd}


def test_oos_degradation_check_passes():
    verdict = oos_degradation_check(_metrics(1.8, 0.6, 8.0), _metrics(1.5, 0.58, 10.0))
    assert verdict["passed"] is True
    assert verdict["reasons"] == []


def test_oos_degradation_check_fails_on_pf_drop():
    verdict = oos_degradation_check(_metrics(2.5, 0.6, 8.0), _metrics(0.9, 0.6, 8.0))
    assert verdict["passed"] is False
    assert any("profit_factor" in c["name"] for c in verdict["checks"] if not c["ok"])
    assert verdict["reasons"]


def test_oos_degradation_check_fails_on_drawdown():
    verdict = oos_degradation_check(_metrics(1.5, 0.6, 5.0), _metrics(1.4, 0.6, 35.0))
    assert verdict["passed"] is False
    names = [c["name"] for c in verdict["checks"] if not c["ok"]]
    assert "oos_drawdown" in names


# ---------------------------------------------------------------------------
# Monte Carlo gate
# ---------------------------------------------------------------------------


def test_mc_gate_skillful_strategy_passes():
    # Consistent small winners: any shuffle is still positive -> beats luck.
    pnls = [0.02] * 60
    gate = monte_carlo_gate(pnls, iterations=5000, seed=7, min_trades=10)
    assert gate["passed"] is True
    assert gate["trades"] == 60
    assert gate["observed_percentile"] > 99.0


def test_mc_gate_coin_flip_fails():
    # Alternating +/- winners: observed ~0 is unremarkable under shuffling.
    pnls = [1.0 if i % 2 == 0 else -1.0 for i in range(60)]
    gate = monte_carlo_gate(pnls, iterations=5000, seed=7, min_trades=10)
    assert gate["passed"] is False
    assert "luck" in gate["note"]


def test_mc_gate_too_few_trades_fails():
    gate = monte_carlo_gate([5.0, 5.0, 5.0], iterations=2000, seed=1,
                            min_trades=10)
    assert gate["passed"] is False
    assert "too few trades" in gate["note"]


def test_mc_gate_deterministic_with_seed():
    pnls = [0.01 * i % 3 - 0.5 for i in range(80)]
    a = monte_carlo_gate(pnls, iterations=3000, seed=11)
    b = monte_carlo_gate(pnls, iterations=3000, seed=11)
    assert a == b
    assert a["engine"] in ("julia", "python")


def test_mc_gate_drawdown_stress():
    pnls = [0.02] * 60
    gate = monte_carlo_gate(pnls, iterations=3000, seed=5, min_trades=10,
                            observed_max_dd_pct=5.0)
    assert gate["passed"] is True
    assert "mc_max_dd_p95" in gate

    bad = monte_carlo_gate(pnls, iterations=3000, seed=5, min_trades=10,
                           observed_max_dd_pct=99.0, max_dd_stress_pct=5.0)
    assert bad["passed"] is False
    assert bad["dd_passed"] is False


def test_mc_max_drawdowns_shape():
    dd = mc_max_drawdowns([1.0, -1.0] * 20, iterations=500, seed=3)
    assert len(dd) == 500
    assert (dd >= 0).all()


@pytest.mark.skipif(not jb.available(), reason="Julia runtime / juliacall not installed")
def test_mc_parity_between_julia_and_python():
    pnls = [0.02, -0.01, 0.03, -0.02, 0.01] * 20
    e_j = jb.mc_equities(pnls, 3000, seed=9, slippage_std=0.1)
    e_p = jb.mc_equities_python(pnls, 3000, seed=9, slippage_std=0.1)

    assert len(e_j) == len(e_p) == 3000
    # Sign-flip null: each draw has zero expectation, so both means must agree
    # within sampling error of the mean (~ std/sqrt(iterations)) — the two
    # engines use independent RNG streams, so an exact match is impossible.
    se = e_p.std() / np.sqrt(len(e_p))
    assert abs(e_j.mean() - e_p.mean()) < 4.0 * se
    # Statistical agreement of the spread (loose tolerance).
    assert abs(e_j.std() - e_p.std()) < 0.15 * e_p.std()


# ---------------------------------------------------------------------------
# Validation log + full runner
# ---------------------------------------------------------------------------


def test_validation_log_roundtrip(tmp_path):
    log = ValidationLog(root=tmp_path)
    log.log({"kind": "monte_carlo", "passed": True})
    log.log({"kind": "monte_carlo", "passed": False})
    entries = log.entries()
    assert len(entries) == 2
    assert entries[0]["run_id"] == log.run_id
    assert entries[0]["passed"] is True
    assert entries[1]["passed"] is False
    assert ValidationLog(root=tmp_path, run_id="nope").entries() == []


def test_run_walk_forward_end_to_end(tmp_path):
    np.random.seed(4)
    n = 1200
    close = 100 + np.cumsum(np.random.normal(0.0005, 0.01, n))
    df = pd.DataFrame({
        "time": np.arange(n),
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.ones(n),
    })

    def _metrics(slice_df):
        c = slice_df["close"].values
        rets = np.diff(c)
        wins = rets[rets > 0].sum()
        losses = -rets[rets < 0].sum()
        return {
            "net_pnl": float(rets.sum()),
            "win_rate": float((rets > 0).mean()),
            "profit_factor": float(wins / losses) if losses > 0 else 999.0,
            "max_dd_pct": 5.0,
            "trades": int(len(rets)),
        }

    result = run_walk_forward(df, _metrics, n_folds=5, train_frac=0.6,
                              purge_bars=10)
    assert len(result["oos_folds"]) == 5
    assert result["aggregate_oos"]["trades"] > 0
    assert "verdict" in result and "passed" in result["verdict"]

    # logged variant writes a JSONL record
    log = ValidationLog(root=tmp_path)
    result2 = run_walk_forward(df, _metrics, log=log)
    assert len(log.entries()) == 1
    assert log.entries()[0]["kind"] == "walk_forward"
