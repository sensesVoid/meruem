"""End-to-end tests for the Parametric GP engine (QiMera.md §6.7 / M3).

Covers the four M3 deliverables:
  1. DEAP-based strongly-typed parametric orchestrator (bounds + int/float kinds).
  2. Julia-backed fitness with the pure-Python twin as fallback (parity asserted).
  3. Hard hurdle gate (min trades / max drawdown) + parsimony penalty.
  4. Single-asset end-to-end run finding a valid champion.
"""

import numpy as np
import pandas as pd
import pytest

from qimera.app import julia_bridge as jb
from qimera.app import gp_engine as ge
from qimera.app.gp_engine import GPConfig, GENE_DEFS, REJECT_SCORE


@pytest.fixture(scope="module")
def ohlcv_df() -> pd.DataFrame:
    """Synthetic single-asset 15m OHLCV (random walk with drift)."""
    np.random.seed(7)
    n = 600
    times = np.arange(1690000000, 1690000000 + n * 900, 900)
    close = 1.09 + np.cumsum(np.random.normal(0.0002, 0.001, n))
    open_p = close + np.random.normal(0, 0.0005, n)
    high = np.maximum(open_p, close) + np.random.uniform(0, 0.001, n)
    low = np.minimum(open_p, close) - np.random.uniform(0, 0.001, n)
    volume = np.random.uniform(10, 1000, n)
    return pd.DataFrame({
        "time": times,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_gp_finds_champion_within_typed_bounds(ohlcv_df):
    """M3 #1+#4: a short single-asset run returns a champion whose genes stay
    inside the typed domains and keep their int/float kinds."""
    cfg = GPConfig(population=20, generations=4, seed=7)
    result = ge.run_gp(ohlcv_df, cfg)

    assert result["best"] is not None
    genes = result["best"]["genes"]
    assert len(genes) == len(GENE_DEFS)

    for gene, definition in zip(genes, GENE_DEFS):
        assert definition["low"] <= gene <= definition["high"]
        if definition["kind"] == "int":
            assert isinstance(gene, int)
        else:
            assert isinstance(gene, float)

    # Deterministic evaluation tally: population × (generations + 1)
    assert result["evals"] == cfg.population * (cfg.generations + 1)
    assert result["engine"] in ("julia", "python")
    assert len(result["hall_of_fame"]) == cfg.hof_size
    assert len(result["logbook"]) == cfg.generations + 1


def test_hurdle_gate_rejects_every_candidate(ohlcv_df):
    """M3 #3: an impossible hurdle (min_trades >> possible) gates every single
    evaluation — no candidate can pass, and the champion carries REJECT_SCORE."""
    cfg = GPConfig(population=10, generations=1, min_trades=10**6, seed=5)
    result = ge.run_gp(ohlcv_df, cfg)

    assert result["evals"] == cfg.population * (cfg.generations + 1)
    assert result["gated"] == result["evals"]
    assert result["best"]["score"] == REJECT_SCORE
    assert result["best"]["metrics"]["gated"] is True


def test_drawdown_gate_filters_deep_drawdowns(ohlcv_df):
    """M3 #3: the drawdown bound is a hard wall — no ACCEPTED candidate may
    exceed it. When every candidate is gated the run reports that honestly
    instead of silently promoting a reject."""
    cfg = GPConfig(population=12, generations=2, max_drawdown_pct=0.1, seed=3)
    result = ge.run_gp(ohlcv_df, cfg)

    best = result["best"]
    assert best is not None
    assert result["accepted"] + result["gated"] == result["evals"]

    if best["metrics"]["gated"]:
        assert best["score"] == REJECT_SCORE
        assert result["accepted"] == 0
    else:
        assert best["metrics"]["max_dd_pct"] <= 0.1
        assert result["accepted"] >= 1


def test_python_fallback_when_julia_absent(ohlcv_df, monkeypatch):
    """M3 #2: without the Julia runtime the engine degrades to the exact
    pure-Python twin and still completes the run."""
    monkeypatch.setattr(ge.jb, "available", lambda: False)
    cfg = GPConfig(population=12, generations=2, seed=3)
    result = ge.run_gp(ohlcv_df, cfg)

    assert result["engine"] == "python"
    assert result["best"] is not None
    assert result["evals"] == cfg.population * (cfg.generations + 1)


@pytest.mark.skipif(not jb.available(), reason="Julia runtime / juliacall not installed")
def test_julia_and_python_scores_agree(ohlcv_df):
    """M3 #2: with Julia present, both engines must score identical metrics —
    the GP population outcome is engine-independent."""
    score_j, metrics_j = ge.evaluate_genes(ohlcv_df, [50, 14, 3.0], engine="julia")
    score_p, metrics_p = ge.evaluate_genes(ohlcv_df, [50, 14, 3.0], engine="python")

    assert metrics_j["net_pnl"] == pytest.approx(metrics_p["net_pnl"], abs=1e-9)
    assert metrics_j["max_dd_pct"] == pytest.approx(metrics_p["max_dd_pct"], abs=1e-9)
    assert metrics_j["trades"] == metrics_p["trades"]
    assert score_j == pytest.approx(score_p, abs=1e-9)


def test_parsimony_penalty_is_exact_gene_bloat(ohlcv_df):
    """M3 #3: parsimony is isolated — for the SAME genes, the score gap between
    weight 0 and weight 1 is exactly the gene-bloat penalty, so the penalty is
    a precise, testable term rather than an approximate bias."""
    genes = [g["high"] for g in GENE_DEFS]
    base = dict(min_trades=0, max_drawdown_pct=1e9)

    score_none, _ = ge.evaluate_genes(
        ohlcv_df, genes, GPConfig(**base, parsimony_weight=0.0))
    score_pen, _ = ge.evaluate_genes(
        ohlcv_df, genes, GPConfig(**base, parsimony_weight=1.0))

    bloat = sum(abs(g - mid) for g, mid in zip(genes, GPConfig()._midpoints))
    assert score_none - score_pen == pytest.approx(bloat, abs=1e-9)
