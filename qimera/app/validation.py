"""QiMera Walk-Forward Validation & Monte Carlo Gate (QiMera.md §6.5 / M4).

Every champion candidate must survive two guardrails before promotion:

1. **Walk-forward validation** — the dataset is split into chronological folds
   with a purged gap between train and out-of-sample test (no leakage). Each
   fold scores OOS and a degradation check compares the in-sample metrics
   against the aggregated OOS metrics with configurable thresholds.

2. **Monte Carlo gate** — the realized trade PnLs are bootstrapped (shuffled
   order + random slippage, Julia-accelerated via ``julia_bridge.mc_equities``
   with an exact pure-Python twin) to answer *"is this better than luck?"*.
   The observed equity must beat the (1-alpha) quantile of the shuffled
   distribution, and the observed drawdown must stay under the MC stress bound.

Every run is appended to a JSONL validation log under ``qimera/data/validation/``
— the M9 champions registry reads these records.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qimera.app import julia_bridge as jb

VALIDATION_ROOT = Path(__file__).resolve().parent.parent / "data" / "validation"

_TF_SECONDS = {
    "m1": 60, "1m": 60, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800,
}

DEFAULT_DEGRADATION_THRESHOLDS: dict = {
    "min_oos_profit_factor": 1.0,
    "min_oos_win_rate": 0.0,
    "max_oos_drawdown_pct": 20.0,
    "max_train_to_oos_pf_drop": 0.5,     # OOS PF >= train PF * (1 - drop)
    "max_train_to_oos_wr_drop_pp": 10.0,  # OOS win rate >= train - this many pp
}


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------


def bars_per_day(timeframe: str) -> int:
    """Approximate bars in one 24h trading day for a timeframe token."""
    seconds = _TF_SECONDS.get(timeframe.lower(), 900)
    return max(1, round(86_400 / seconds))


def walk_forward_folds(n: int, n_folds: int = 5, train_frac: float = 0.6,
                       purge_bars: int = 1, min_train: int = 100,
                       min_test: int = 20) -> list[dict]:
    """Chronological walk-forward fold boundaries over ``n`` bars.

    Each fold is ``{"train": (a, b), "test": (c, d)}`` with half-open index
    ranges: expanding train ``[0, b)``, a ``purge_bars`` gap, then the OOS test
    block ``[c, d)`` strictly after the gap — no leakage across folds.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    gap = max(1, int(purge_bars))
    test_len = max(min_test, int(n * (1.0 - train_frac) / n_folds))

    folds: list[dict] = []
    for i in range(n_folds):
        # Contiguous OOS blocks covering the tail; only the train->test seam
        # carries the purged gap (no leakage into the test window).
        test_end = n - (n_folds - 1 - i) * test_len
        test_start = test_end - test_len
        train_end = test_start - gap
        if train_end < min_train:
            raise ValueError(
                f"not enough bars for fold {i}: train would be {train_end} bars "
                f"(min {min_train}); need more data or fewer folds")
        folds.append({"train": (0, train_end), "test": (test_start, test_end)})
    return folds


# ---------------------------------------------------------------------------
# OOS degradation check
# ---------------------------------------------------------------------------


def oos_degradation_check(train_metrics: dict, oos_metrics: dict,
                          thresholds: dict | None = None) -> dict:
    """Compare in-sample vs out-of-sample metrics; pass/fail with per-metric checks.

    Metric keys follow the Julia ``fitness`` contract: ``profit_factor``,
    ``win_rate``, ``max_dd_pct`` (positive drawdown percentage).
    """
    t = {**(thresholds or DEFAULT_DEGRADATION_THRESHOLDS)}
    train_pf = float(train_metrics.get("profit_factor") or 0.0)
    train_wr = float(train_metrics.get("win_rate") or 0.0)
    oos_pf = float(oos_metrics.get("profit_factor") or 0.0)
    oos_wr = float(oos_metrics.get("win_rate") or 0.0)
    oos_dd = float(oos_metrics.get("max_dd_pct") or 0.0)

    checks = [
        {
            "name": "oos_profit_factor",
            "ok": oos_pf >= float(t["min_oos_profit_factor"]),
            "detail": f"OOS PF {oos_pf:.3f} < {t['min_oos_profit_factor']:.2f}",
        },
        {
            "name": "oos_pf_degradation",
            "ok": train_pf <= 0 or oos_pf >= train_pf * (1.0 - float(t["max_train_to_oos_pf_drop"])),
            "detail": (f"OOS PF {oos_pf:.3f} degraded >{float(t['max_train_to_oos_pf_drop'])*100:.0f}% "
                       f"vs train {train_pf:.3f}"),
        },
        {
            "name": "oos_win_rate_degradation",
            "ok": oos_wr >= train_wr - float(t["max_train_to_oos_wr_drop_pp"]) / 100.0,
            "detail": (f"OOS win rate {oos_wr:.3f} dropped >"
                       f"{t['max_train_to_oos_wr_drop_pp']:.0f}pp vs train {train_wr:.3f}"),
        },
        {
            "name": "oos_drawdown",
            "ok": oos_dd <= float(t["max_oos_drawdown_pct"]),
            "detail": f"OOS max DD {oos_dd:.2f}% > {t['max_oos_drawdown_pct']:.1f}%",
        },
    ]
    for c in checks:
        c["detail"] = "" if c["ok"] else c["detail"]
    reasons = [c["detail"] for c in checks if not c["ok"]]
    return {"passed": not reasons, "checks": checks, "reasons": reasons}


# ---------------------------------------------------------------------------
# Monte Carlo gate
# ---------------------------------------------------------------------------


def monte_carlo_gate(trade_pnls, iterations: int = 10_000, seed: int = 42,
                     slippage_std: float = 0.0, latency_bars: int = 0,
                     alpha: float = 0.05, min_trades: int = 10,
                     observed_max_dd_pct: float | None = None,
                     max_dd_stress_pct: float = 20.0) -> dict:
    """Bootstrap stress gate: does the observed equity beat shuffled luck?

    The realized PnL sequence is bootstrapped ``iterations`` times (each trade's
    sign randomly flipped — the classic edge-vs-luck null, since a plain order
    shuffle would leave the equity sum invariant — with Gaussian slippage added,
    Julia-accelerated when available).
    ``latency_bars`` is folded into effective slippage (each delayed bar adds a
    pro-rata cost component) to keep the model deterministic and single-seed.

    The gate passes when the observed total equity is at or above the
    ``(1 - alpha)`` quantile of the shuffled distribution, the trade count is
    sufficient, and (when ``observed_max_dd_pct`` is given) the observed
    drawdown stays under the MC stress bound.
    """
    pnls = np.asarray([float(p) for p in trade_pnls])
    n = int(len(pnls))
    if n == 0:
        return {"passed": False, "iterations": iterations, "trades": 0,
                "note": "no trades to bootstrap"}

    eff_slippage = float(slippage_std) * (1.0 + 0.1 * max(0, int(latency_bars)))
    engine = "julia" if jb.available() else "python"
    if engine == "julia":
        equities = jb.mc_equities(pnls.tolist(), iterations, seed, eff_slippage)
    else:
        equities = jb.mc_equities_python(pnls.tolist(), iterations, seed, eff_slippage)
    equities = np.asarray(equities, dtype=float)

    observed = float(pnls.sum())
    quantile = float(np.quantile(equities, 1.0 - alpha))
    percentile = float((equities < observed).mean() * 100.0)

    result = {
        "passed": bool(observed >= quantile and n >= min_trades),
        "trades": n,
        "observed_equity": observed,
        "iterations": iterations,
        "mc_mean": float(equities.mean()),
        "mc_std": float(equities.std()),
        "mc_p5": float(np.quantile(equities, 0.05)),
        "mc_p50": float(np.quantile(equities, 0.50)),
        "mc_p95": float(np.quantile(equities, 0.95)),
        "observed_percentile": percentile,
        "alpha_quantile": quantile,
        "engine": engine,
        "note": "",
    }

    dd_stress = None
    if observed_max_dd_pct is not None:
        dd_stress = mc_max_drawdowns(pnls, iterations=min(iterations, 2000),
                                     seed=seed, slippage_std=eff_slippage)
        dd_p95 = float(np.quantile(dd_stress, 0.95))
        result["mc_max_dd_p95"] = dd_p95
        result["observed_max_dd_pct"] = observed_max_dd_pct
        result["dd_passed"] = bool(observed_max_dd_pct <= max(dd_p95, max_dd_stress_pct))
        result["passed"] = result["passed"] and result["dd_passed"]

    if not result["passed"]:
        if n < min_trades:
            result["note"] = f"too few trades ({n} < {min_trades})"
        elif observed < quantile:
            result["note"] = "equity does not beat shuffled luck"
        elif dd_stress is not None and not result.get("dd_passed", True):
            result["note"] = "observed drawdown exceeds the MC stress bound"
    return result


def mc_max_drawdowns(trade_pnls, iterations: int = 2000, seed: int = 42,
                     slippage_std: float = 0.0) -> np.ndarray:
    """Max-drawdown distribution from sign-flipped trade sequences (numpy twin)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray([float(p) for p in trade_pnls])
    n = int(len(arr))
    out = np.empty(iterations)
    for it in range(iterations):
        seq = arr[rng.permutation(n)]
        if slippage_std > 0.0:
            seq = seq + rng.normal(0.0, slippage_std, n)
        eq = np.cumsum(seq)
        peak = np.maximum.accumulate(eq)
        dd = np.where(peak > 0, (peak - eq) / np.maximum(peak, 1e-12) * 100.0, 0.0)
        out[it] = float(dd.max()) if len(dd) else 0.0
    return out


# ---------------------------------------------------------------------------
# Full walk-forward runner + validation log
# ---------------------------------------------------------------------------


class ValidationLog:
    """Append-only JSONL log of validation runs under ``qimera/data/validation``."""

    def __init__(self, root: Path = VALIDATION_ROOT, run_id: str | None = None):
        self.root = Path(root)
        self.run_id = run_id or uuid.uuid4().hex[:12]

    @property
    def path(self) -> Path:
        return self.root / f"{self.run_id}.jsonl"

    def log(self, record: dict) -> dict:
        record = {"run_id": self.run_id,
                  "ts": datetime.now(timezone.utc).isoformat(), **record}
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return record

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _average_metrics(metrics_list: list[dict]) -> dict:
    keys = ("net_pnl", "win_rate", "profit_factor", "max_dd_pct", "trades")
    out: dict = {}
    for k in keys:
        vals = [float(m.get(k) or 0.0) for m in metrics_list if m]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


def run_walk_forward(df, metrics_fn, n_folds: int = 5, train_frac: float = 0.6,
                     purge_bars: int = 1, thresholds: dict | None = None,
                     log: ValidationLog | None = None) -> dict:
    """Full walk-forward run: folds -> per-fold OOS metrics -> degradation verdict.

    ``metrics_fn(DataFrame)`` returns the Julia-fitness-contract metric dict
    (``profit_factor``/``win_rate``/``max_dd_pct`` at minimum).
    """
    n = len(df)
    folds = walk_forward_folds(n, n_folds, train_frac, purge_bars)
    train_slice = df.iloc[folds[0]["train"][0]:folds[0]["train"][1]]
    train_metrics = metrics_fn(train_slice)

    oos_folds = []
    for f in folds:
        test_slice = df.iloc[f["test"][0]:f["test"][1]]
        oos_folds.append({"train": f["train"], "test": f["test"],
                          "metrics": metrics_fn(test_slice)})
    aggregate_oos = _average_metrics([o["metrics"] for o in oos_folds])
    verdict = oos_degradation_check(train_metrics, aggregate_oos, thresholds)

    record = {"kind": "walk_forward", "n_folds": n_folds,
              "purge_bars": purge_bars, "train_metrics": train_metrics,
              "oos_folds": oos_folds, "aggregate_oos": aggregate_oos,
              "verdict": verdict}
    if log is not None:
        log.log(record)
    return record
