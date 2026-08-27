"""QiMera Multi-Indicator GP Framework (QiMera.md §6.7 / M5).

Extends the M3 parametric GP (``gp_engine.py``) from the single EMA-cross +
ATR-stop family to a **typed multi-indicator strategy family**: VWAP, Fair
Value Gaps, Liquidity levels, Order Blocks and Cumulative Volume Delta.

Each indicator is described by a typed schema (``INDICATOR_SCHEMAS``): an id,
a name, and three typed parameters (int/float kind, [low, high] domain, default).
The schema is the single source of truth for DEAP gene bounds, mutation sigmas
and the parsimony midpoint — M7 (agentic synthesis) consumes the same registry.

The **fitness hook** (``strategy_metrics``) computes the indicator's normalized
long-bias signal series ([-1, +1], positive = bullish) then simulates the
strategy with the M3-style ATR trailing-stop exit, returning the standard
Julia-fitness metric contract ``[net, win_rate, pf, max_dd, trades]``.

The GP genotype is a fixed-length vector:

    [indicator_id:int, p0, p1, p2, atr_len:int, atr_mult:float]

where ``p0..p2`` are the three parameters of the *chosen* indicator (bounds and
int/float kind come from its schema, applied by ``_fix`` after every operator)
and ``atr_len``/``atr_mult`` are the shared exit parameters.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qimera.app.feature_library.primitives import compute_atr

# Score handed to any candidate that trips the hurdle gate (M3 convention).
REJECT_SCORE = -1e12

# Exit-parameter domains shared by every indicator (M3 defaults).
ATR_LEN_DOMAIN = (5, 60)
ATR_MULT_DOMAIN = (0.5, 6.0)

# ---------------------------------------------------------------------------
# Typed indicator schemas (M5 spec: VWAP / FVG / Liquidity / Order Blocks / CVD)
# ---------------------------------------------------------------------------

INDICATOR_SCHEMAS: dict[str, dict[str, Any]] = {
    "vwap": {
        "name": "Session VWAP mean-reversion",
        "description": "Reversion to an anchored VWAP: long bias while price sits "
                       "below the VWAP within a width threshold and is reclaiming.",
        "params": [
            {"name": "width_atr", "kind": "float", "low": 0.2, "high": 3.0, "default": 1.0},
            {"name": "min_bars", "kind": "int", "low": 5, "high": 200, "default": 48},
            {"name": "reclaim_bars", "kind": "int", "low": 1, "high": 5, "default": 1},
        ],
    },
    "fvg": {
        "name": "Fair Value Gap retest",
        "description": "Long bias after an unmitigated bullish FVG forms, decaying "
                       "with freshness and invalidated when price fills the gap.",
        "params": [
            {"name": "min_gap_atr", "kind": "float", "low": 0.1, "high": 2.0, "default": 0.3},
            {"name": "expiry_bars", "kind": "int", "low": 5, "high": 100, "default": 30},
            {"name": "invalidate_bars", "kind": "int", "low": 1, "high": 10, "default": 3},
        ],
    },
    "liquidity": {
        "name": "Liquidity sweep reversal",
        "description": "Long bias after sell-side liquidity is swept (price pierces "
                       "below the rolling low by a tolerance) and closes back inside.",
        "params": [
            {"name": "lookback", "kind": "int", "low": 5, "high": 60, "default": 20},
            {"name": "eq_tol_atr", "kind": "float", "low": 0.0, "high": 0.5, "default": 0.1},
            {"name": "confirm_bars", "kind": "int", "low": 1, "high": 5, "default": 1},
        ],
    },
    "order_blocks": {
        "name": "Order Block proximity",
        "description": "Long bias when price is near an unmitigated bullish order "
                       "block (high-volume engulfing bar), with freshness decay.",
        "params": [
            {"name": "vol_period", "kind": "int", "low": 10, "high": 60, "default": 20},
            {"name": "freshness", "kind": "int", "low": 10, "high": 150, "default": 80},
            {"name": "mitigation_atr", "kind": "float", "low": 0.1, "high": 3.0, "default": 1.0},
        ],
    },
    "cvd": {
        "name": "Cumulative Volume Delta momentum",
        "description": "Long bias when the session-normalized CVD z-score breaks "
                       "above the delta threshold, smoothed over confirm bars.",
        "params": [
            {"name": "delta_atr", "kind": "float", "low": 0.5, "high": 10.0, "default": 2.0},
            {"name": "profile_period", "kind": "int", "low": 5, "high": 60, "default": 20},
            {"name": "confirm_bars", "kind": "int", "low": 1, "high": 5, "default": 1},
        ],
    },
}

INDICATOR_IDS: list[str] = list(INDICATOR_SCHEMAS)


@dataclass
class GPConfig:
    """Knobs for the multi-indicator GP run (M5)."""

    population: int = 40
    generations: int = 12
    cxpb: float = 0.6
    mutpb: float = 0.3
    tournament_size: int = 3
    hof_size: int = 5

    # Hurdle gate
    min_trades: int = 5
    max_drawdown_pct: float = 15.0

    # Soft terms
    dd_free_band: float = 5.0
    dd_penalty_weight: float = 0.02
    parsimony_weight: float = 0.0005

    entry_threshold: float = 0.0   # signal must cross above this to go long
    indicator_switch_pb: float = 0.15  # per-mutation probability of switching family

    seed: int = 42
    _midpoints: dict[str, list[float]] = field(
        default_factory=lambda: {
            ind: [(p["low"] + p["high"]) / 2.0 for p in s["params"]]
            for ind, s in INDICATOR_SCHEMAS.items()
        },
        init=False, repr=False,
    )


def _as_df(df) -> pd.DataFrame:
    """Normalize the input to a DataFrame with the OHLCV contract."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("multi-indicator GP data must be a pandas DataFrame")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"gp data missing required column '{col}'")
    return df


def _atr_array(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder ATR as a float array; NaN/<=0 entries fall back to 1.0.

    ``period`` is honoured so the ``atr_len`` gene actually controls the exit
    stop (a fixed 14 here would silently neuter that gene). Leading NaN is
    replaced with a constant, never a future value (no lookahead).
    """
    atr = compute_atr(df["high"], df["low"], df["close"], period).astype(float)
    atr = atr.replace([np.inf, -np.inf], np.nan)
    arr = atr.to_numpy()
    return np.where(np.isfinite(arr) & (arr > 0), arr, 1.0)


def _session_ids(df: pd.DataFrame) -> np.ndarray:
    """Session ids: UTC day rollover when a ``time`` column exists, else all-ones.

    Mirrors the 'session resets' requirement of the VWAP schema: each new UTC day
    re-anchors the VWAP (and the CVD accumulation).
    """
    if "time" not in df.columns:
        return np.zeros(len(df), dtype=int)
    t = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    if t.isna().all():
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    day = t.dt.normalize()
    return (day != day.shift(1).fillna(day.iloc[0])).cumsum().to_numpy()


# ---------------------------------------------------------------------------
# Per-indicator signal functions (all -> normalized [-1, +1] long-bias series)
# ---------------------------------------------------------------------------


def _signal_vwap(df: pd.DataFrame, width_atr: float, min_bars: int,
                 reclaim_bars: int) -> pd.Series:
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    v = df["volume"].to_numpy(dtype=float)
    n = len(c)
    atr = _atr_array(df)
    session = _session_ids(df)

    tp = (h + l + c) / 3.0
    cum_pv = np.zeros(n)
    cum_v = np.zeros(n)
    cur_pv = cur_v = 0.0
    for i in range(n):
        if i == 0 or session[i] != session[i - 1]:
            cur_pv = cur_v = 0.0
        cur_pv += tp[i] * v[i]
        cur_v += v[i]
        cum_pv[i], cum_v[i] = cur_pv, cur_v
    vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-12), tp)

    dist = (c - vwap) / np.maximum(atr, 1e-12)          # in ATR units
    raw = -np.tanh(dist / max(float(width_atr), 1e-9))  # below vwap -> bullish
    band = np.where(np.abs(dist) < float(width_atr) * 2.0, raw, 0.0)

    out = pd.Series(band).rolling(max(int(reclaim_bars), 1), min_periods=1).max()
    out.iloc[: max(int(min_bars), 1)] = 0.0              # warmup: anchor not trusted
    return out.fillna(0.0)


def _signal_fvg(df: pd.DataFrame, min_gap_atr: float, expiry_bars: int,
                invalidate_bars: int) -> pd.Series:
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    n = len(c)
    atr = _atr_array(df)
    expiry = max(int(expiry_bars), 1)
    invalidate = max(int(invalidate_bars), 1)
    min_gap = max(float(min_gap_atr), 0.0)

    out = np.zeros(n)
    bull: tuple[float, int] | None = None   # (gap floor, formed bar)
    bear: tuple[float, int] | None = None
    for i in range(2, n):
        # Expiration + fill invalidation of existing zones (grace of invalidate bars).
        if bull is not None:
            if i - bull[1] >= expiry or c[i] < bull[0]:
                bull = None
        if bear is not None:
            if i - bear[1] >= expiry or c[i] > bear[0]:
                bear = None
        # New gap: 3-candle imbalance (bar i low vs bar i-2 high), ATR-scaled.
        gap = l[i] - h[i - 2]
        if gap > 0 and gap / atr[i] >= min_gap and bull is None:
            bull = (h[i - 2], i)
        bear_gap = l[i - 2] - h[i]
        if bear_gap > 0 and bear_gap / atr[i] >= min_gap and bear is None:
            bear = (l[i - 2], i)
        # Signal with freshness decay.
        if bull is not None:
            out[i] = 1.0 - (i - bull[1]) / expiry
        elif bear is not None:
            out[i] = -(1.0 - (i - bear[1]) / expiry)
    return pd.Series(out)


def _signal_liquidity(df: pd.DataFrame, lookback: int, eq_tol_atr: float,
                      confirm_bars: int) -> pd.Series:
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    n = len(c)
    atr = _atr_array(df)
    look = max(int(lookback), 3)
    tol = float(eq_tol_atr)
    confirm = max(int(confirm_bars), 1)

    out = np.zeros(n)
    for i in range(look, n):
        win_h = h[i - look:i]
        win_l = l[i - look:i]
        max_h = win_h.max()
        min_l = win_l.min()
        if l[i] < min_l - tol * atr[i] and c[i] > min_l:   # sell-side sweep, reclaim
            out[i] = 1.0
        elif h[i] > max_h + tol * atr[i] and c[i] < max_h:  # buy-side sweep, rejection
            out[i] = -1.0
    s = pd.Series(out)
    pos = s.rolling(confirm, min_periods=1).max()
    neg = s.rolling(confirm, min_periods=1).min()
    return pd.Series(np.where(pos > 0, pos, neg))


def _signal_order_blocks(df: pd.DataFrame, vol_period: int, freshness: int,
                         mitigation_atr: float) -> pd.Series:
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    v = df["volume"].to_numpy(dtype=float)
    n = len(c)
    atr = _atr_array(df)
    vol_ma = pd.Series(v).rolling(max(int(vol_period), 2), min_periods=1).mean().to_numpy()
    fresh = max(int(freshness), 5)
    mit = float(mitigation_atr)

    zones: list[dict] = []
    out = np.zeros(n)
    for i in range(1, n):
        for z in zones:
            z["age"] += 1
        # Mitigation (price through the zone past the ATR tolerance) + freshness.
        zones = [
            z for z in zones
            if z["age"] <= fresh
            and not (z["kind"] == "bull" and c[i] < z["level"] - mit * atr[i])
            and not (z["kind"] == "bear" and c[i] > z["level"] + mit * atr[i])
        ]
        # New zone: high-volume engulfing bar.
        if v[i] > 1.5 * vol_ma[i]:
            if c[i] > h[i - 1] and c[i - 1] < c[i]:
                zones.append({"kind": "bull", "level": l[i - 1], "age": 0})
            elif c[i] < l[i - 1] and c[i - 1] > c[i]:
                zones.append({"kind": "bear", "level": h[i - 1], "age": 0})
        # Proximity bias to the nearest active zone.
        if zones:
            best = min(zones, key=lambda z: abs(c[i] - z["level"]))
            dist = (c[i] - best["level"]) / atr[i]
            decay = 1.0 - best["age"] / fresh
            span = max(abs(dist) / max(mit, 1e-9), 1e-9)
            if best["kind"] == "bull" and -mit < dist < 0.5:
                out[i] = decay * max(0.0, 1.0 - span)
            elif best["kind"] == "bear" and -0.5 < dist < mit:
                out[i] = -decay * max(0.0, 1.0 - span)
    return pd.Series(out)


def _signal_cvd(df: pd.DataFrame, delta_atr: float, profile_period: int,
                confirm_bars: int) -> pd.Series:
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    v = df["volume"].to_numpy(dtype=float)
    n = len(c)
    session = _session_ids(df)

    rng = np.maximum(h - l, 1e-12)
    buy = v * np.clip((c - l) / rng, 0.0, 1.0)
    delta = 2.0 * buy - v                      # signed volume delta
    cvd = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if i == 0 or session[i] != session[i - 1]:
            cur = 0.0
        cur += delta[i]
        cvd[i] = cur

    prof = max(int(profile_period), 2)
    cvd_ma = pd.Series(cvd).rolling(prof, min_periods=1).mean().to_numpy()
    cvd_std = pd.Series(cvd).rolling(prof, min_periods=1).std().to_numpy()
    cvd_std = np.where(np.isfinite(cvd_std) & (cvd_std > 0), cvd_std, 1.0)
    z = (cvd - cvd_ma) / cvd_std
    sig = np.tanh(z / max(float(delta_atr), 0.1))
    return pd.Series(sig).rolling(max(int(confirm_bars), 1), min_periods=1).mean().fillna(0.0)


_SIGNAL_FNS = {
    "vwap": _signal_vwap,
    "fvg": _signal_fvg,
    "liquidity": _signal_liquidity,
    "order_blocks": _signal_order_blocks,
    "cvd": _signal_cvd,
}


def signal_for(indicator_id: str, df, params: list) -> pd.Series:
    """Compute the normalized long-bias signal series for an indicator+params."""
    fn = _SIGNAL_FNS.get(indicator_id)
    if fn is None:
        raise ValueError(f"unknown indicator '{indicator_id}'; "
                         f"expected one of {list(INDICATOR_SCHEMAS)}")
    schema = INDICATOR_SCHEMAS[indicator_id]
    p = [float(x) for x in params]
    if len(p) != len(schema["params"]):
        raise ValueError(
            f"indicator '{indicator_id}' expects {len(schema['params'])} params, "
            f"got {len(p)}")
    return fn(_as_df(df), *p)


# ---------------------------------------------------------------------------
# Fitness hook: normalized signal -> ATR trailing-stop simulation (M3 contract)
# ---------------------------------------------------------------------------


def strategy_metrics(df, indicator_id: str, params: list,
                     atr_len: int = 14, atr_mult: float = 2.0,
                     entry_threshold: float = 0.0) -> list[float]:
    """Simulate the indicator strategy and return the Julia fitness contract.

    ``[net_pnl, win_rate, profit_factor, max_drawdown_pct, trades]`` — the exact
    shape the M3 gate, walk-forward layer (M4) and champions registry (M9) consume.
    """
    signal = signal_for(indicator_id, df, params).to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    atr_len_i = max(int(atr_len), 2)
    atr = _atr_array(df, period=atr_len_i)   # the atr_len gene drives the exit
    n = len(c)
    atr_mult_f = max(float(atr_mult), 0.1)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    trades = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    entry = 0.0
    stop = 0.0
    for i in range(1, n):
        if entry == 0.0 and signal[i - 1] <= entry_threshold < signal[i]:
            entry = c[i]
            stop = entry - atr_mult_f * atr[i]
            continue
        if entry != 0.0:
            cand = c[i] - atr_mult_f * atr[i]
            if cand > stop:
                stop = cand
            if c[i] < stop:
                pnl = c[i] - entry
                equity += pnl
                trades += 1
                if pnl >= 0:
                    wins += 1
                    gross_win += pnl
                else:
                    gross_loss -= pnl
                entry = 0.0
        if equity > peak:
            peak = equity
        elif peak > 0.0:
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    if gross_loss > 0.0:
        pf = min(gross_win / gross_loss, 1.0e9)
    elif gross_win > 0.0:
        pf = 1.0e9
    else:
        pf = 0.0
    win_rate = wins / trades if trades > 0 else 0.0
    return [equity, win_rate, pf, max_dd, float(trades)]


def _parsimony_penalty(indicator_id: str, params: list) -> float:
    """Drift of the active indicator's params from their domain midpoints."""
    mids = GPConfig()._midpoints[indicator_id]
    return sum(abs(p - m) for p, m in zip(params, mids))


def evaluate_genes(df, genes: list, cfg: GPConfig | None = None,
                   engine: str | None = None) -> tuple[float, dict]:
    """Score one multi-indicator genotype (public, used by tests/UI).

    ``engine`` is accepted for API compatibility with ``gp_engine.evaluate_genes``;
    the M5 simulator is pure Python (the signal layer is pandas/numpy), so the
    engine is always ``"python"``.
    """
    cfg = cfg or GPConfig()
    ind_id = INDICATOR_IDS[int(round(genes[0])) % len(INDICATOR_IDS)]
    params = [float(x) for x in genes[1:4]]
    atr_len, atr_mult = int(genes[4]), float(genes[5])

    net, win_rate, pf, max_dd, trades = strategy_metrics(
        _as_df(df), ind_id, params, atr_len=atr_len, atr_mult=atr_mult,
        entry_threshold=cfg.entry_threshold)

    # Coerce to plain floats: numpy indexing yields np.float64 values that would
    # break JSON serialization downstream (M4 validation log / M10 manifest).
    net, win_rate, pf, max_dd = (float(x) for x in (net, win_rate, pf, max_dd))
    trades = int(trades)

    if trades < cfg.min_trades or max_dd > cfg.max_drawdown_pct:
        return REJECT_SCORE, {
            "indicator": ind_id, "genes": list(genes), "net_pnl": net,
            "win_rate": win_rate, "profit_factor": pf, "max_dd_pct": max_dd,
            "trades": trades, "gated": True, "score": REJECT_SCORE,
        }

    bloat = _parsimony_penalty(ind_id, params)
    dd_penalty = cfg.dd_penalty_weight * max(0.0, max_dd - cfg.dd_free_band)
    score = float(net) - dd_penalty - cfg.parsimony_weight * bloat

    return score, {
        "indicator": ind_id, "genes": list(genes), "net_pnl": net,
        "win_rate": win_rate, "profit_factor": pf, "max_dd_pct": max_dd,
        "trades": trades, "gated": False, "score": score,
    }


# ---------------------------------------------------------------------------
# DEAP orchestration (mirrors the M3 loop, adds categorical indicator switching)
# ---------------------------------------------------------------------------


def _fix(individual: list, cfg: GPConfig) -> list:
    """Restore per-gene bounds and int/float kinds after variation.

    The p0..p2 bounds depend on the *current* indicator gene, so the switch
    operator must fix after every mutation/crossover.
    """
    individual[0] = int(round(individual[0])) % len(INDICATOR_IDS)
    schema = INDICATOR_SCHEMAS[INDICATOR_IDS[individual[0]]]
    for k, p in enumerate(schema["params"]):
        v = individual[1 + k]
        if p["kind"] == "int":
            v = int(round(v))
        individual[1 + k] = min(max(v, p["low"]), p["high"])
    individual[4] = int(round(individual[4]))
    individual[4] = min(max(individual[4], ATR_LEN_DOMAIN[0]), ATR_LEN_DOMAIN[1])
    individual[5] = round(float(individual[5]), 3)
    individual[5] = min(max(individual[5], ATR_MULT_DOMAIN[0]), ATR_MULT_DOMAIN[1])
    return individual


def run_multi_gp(df, cfg: GPConfig | None = None) -> dict:
    """Run the multi-indicator GP and return the champion summary.

    ``df`` is a DataFrame with the OHLCV contract (``open``/``high``/``low``/
    ``close``/``volume``). Returns a JSON-safe summary with the hall of fame,
    evaluation tally, compact logbook and the indicator schema reference.

    Note: DEAP draws from the process-global ``random`` module, so the run
    re-seeds it for reproducibility (same caveat as ``gp_engine.run_gp``).
    """
    cfg = cfg or GPConfig()
    df = _as_df(df)

    random.seed(cfg.seed)

    from deap import base, creator, tools
    from deap.algorithms import varAnd

    if not hasattr(creator, "FitnessMultiGP"):
        creator.create("FitnessMultiGP", base.Fitness, weights=(1.0,))
        creator.create("IndividualMultiGP", list, fitness=creator.FitnessMultiGP)

    toolbox = base.Toolbox()

    def _attr_indicator():
        return random.randrange(len(INDICATOR_IDS))

    def _attr_param(kind: str):
        if kind == "int":
            return random.randint(0, 200)
        return round(random.uniform(0.0, 6.0), 3)

    toolbox.register("attr_indicator", _attr_indicator)
    for k in range(3):
        toolbox.register(f"attr_p{k}", lambda k=k: _attr_param("float"))
    toolbox.register("attr_atr_len", lambda: random.randint(*ATR_LEN_DOMAIN))
    toolbox.register("attr_atr_mult", lambda: round(random.uniform(*ATR_MULT_DOMAIN), 3))

    def _individual():
        return creator.IndividualMultiGP([
            toolbox.attr_indicator(),
            toolbox.attr_p0(), toolbox.attr_p1(), toolbox.attr_p2(),
            toolbox.attr_atr_len(), toolbox.attr_atr_mult(),
        ])

    toolbox.register("individual", _individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def _mate(a, b):
        tools.cxBlend(a, b, alpha=0.5)
        return _fix(a, cfg), _fix(b, cfg)

    def _mutate(ind):
        if random.random() < cfg.indicator_switch_pb:
            ind[0] = random.randrange(len(INDICATOR_IDS))
        schema = INDICATOR_SCHEMAS[INDICATOR_IDS[int(round(ind[0])) % len(INDICATOR_IDS)]]
        for k, p in enumerate(schema["params"]):
            if random.random() < 0.4:
                span = (p["high"] - p["low"]) / 10.0
                ind[1 + k] += random.gauss(0.0, span)
        if random.random() < 0.4:
            ind[4] += random.gauss(0.0, 4.0)
        if random.random() < 0.4:
            ind[5] += random.gauss(0.0, 0.5)
        return (_fix(ind, cfg),)

    toolbox.register("mate", _mate)
    toolbox.register("mutate", _mutate)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_size)

    tally = {"evals": 0, "gated": 0, "accepted": 0}

    def _deap_evaluate(ind):
        score, metrics = evaluate_genes(df, ind, cfg)
        tally["evals"] += 1
        if metrics["gated"]:
            tally["gated"] += 1
        else:
            tally["accepted"] += 1
        ind.metrics = metrics
        return (score,)

    toolbox.register("evaluate", _deap_evaluate)

    pop = toolbox.population(n=cfg.population)
    hof = tools.HallOfFame(cfg.hof_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    hof.update(pop)
    logbook.record(gen=0, nevals=len(pop), **stats.compile(pop))

    for gen in range(1, cfg.generations + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(o) for o in offspring]
        offspring = varAnd(offspring, toolbox, cfg.cxpb, cfg.mutpb)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = tools.selBest(offspring, len(pop))
        hof.update(pop)
        logbook.record(gen=gen, nevals=len(offspring), **stats.compile(pop))

    best = hof[0] if len(hof) else None
    return {
        "engine": "python",
        "indicators": [
            {"id": iid, "name": INDICATOR_SCHEMAS[iid]["name"],
             "params": INDICATOR_SCHEMAS[iid]["params"]}
            for iid in INDICATOR_IDS
        ],
        "best": (
            {
                "indicator": INDICATOR_IDS[int(round(best[0])) % len(INDICATOR_IDS)],
                "genes": list(best),
                "score": float(best.fitness.values[0]),
                "metrics": dict(best.metrics),
            }
            if best is not None
            else None
        ),
        "hall_of_fame": [
            {
                "indicator": INDICATOR_IDS[int(round(ind[0])) % len(INDICATOR_IDS)],
                "genes": list(ind),
                "score": float(ind.fitness.values[0]),
            }
            for ind in hof
        ],
        "evals": tally["evals"],
        "gated": tally["gated"],
        "accepted": tally["accepted"],
        # DEAP's stats.compile yields numpy scalars; coerce to plain floats so
        # the summary stays JSON-safe across numpy versions.
        "logbook": [
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in row.items()}
            for row in logbook
        ],
    }
