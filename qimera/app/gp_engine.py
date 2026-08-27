"""QiMera Parametric GP Orchestrator (QiMera.md §6.7 / M3).

Single-asset, single-indicator parametric genetic programming over the
EMA-cross + ATR trailing-stop strategy family — the exact family the Julia
``fitness`` (qimera/julia/src/QiMera.jl) simulates.

Individuals are *typed* parameter vectors:

    [ema_len:int, atr_len:int, atr_mult:float]

"Strongly typed" here means every gene has its own [low, high] domain and its
int/float kind, and every operator (init / blend-crossover / gaussian-mutation)
preserves both — ints stay ints, floats stay floats, and values stay inside
bounds.

DEAP drives the population loop; the Julia bridge scores each individual
(``qimera.app.julia_bridge.fitness``) with the pure-Python twin as an exact
fallback when the Julia runtime is absent — same graceful-degradation contract
as the rest of QiMera.

Two guardrails from the M3 spec are enforced inside the score:

- **Hard hurdle gate** — candidates below ``min_trades`` or above
  ``max_drawdown_pct`` get a reject score and can never win selection.
- **Parsimony penalty** — a per-unit cost on genes drifting from their domain
  midpoint, the parametric analogue of tree-bloat control.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qimera.app import julia_bridge as jb

# Typed gene definitions — order MUST match the Julia
# ``fitness(close, high, low, ema_len, atr_len, atr_mult)`` signature.
GENE_DEFS: list[dict[str, Any]] = [
    {"name": "ema_len", "low": 5, "high": 200, "kind": "int"},
    {"name": "atr_len", "low": 2, "high": 60, "kind": "int"},
    {"name": "atr_mult", "low": 0.5, "high": 6.0, "kind": "float"},
]

# Score handed to any candidate that trips the hurdle gate: deterministic,
# hugely negative, so a gated candidate can never out-rank a valid one.
REJECT_SCORE = -1e12


@dataclass
class GPConfig:
    """Knobs for the parametric GP run (M3)."""

    population: int = 40
    generations: int = 12
    cxpb: float = 0.6            # per-pair crossover probability
    mutpb: float = 0.3           # per-individual mutation probability
    tournament_size: int = 3
    hof_size: int = 5

    # Hurdle gate (hard rejection bounds on the Julia fitness metrics)
    min_trades: int = 5          # reject strategies with fewer closed trades
    max_drawdown_pct: float = 15.0  # reject strategies whose max DD exceeds this (%)

    # Soft terms on top of the gate
    dd_free_band: float = 5.0        # drawdown % allowed with no penalty
    dd_penalty_weight: float = 0.02  # per-% soft penalty above the free band
    parsimony_weight: float = 0.0005  # per-unit penalty for gene "bloat"

    seed: int = 42
    # DEAP uses these lazily (imported inside run), so dataclass field() marks
    # them as non-init to keep GPConfig JSON-constructible.
    _midpoints: list[float] = field(
        default_factory=lambda: [(g["low"] + g["high"]) / 2.0 for g in GENE_DEFS],
        init=False, repr=False,
    )


def _as_arrays(df) -> tuple[list[float], list[float], list[float]]:
    """Pull close/high/low out of the M1 data-layer DataFrame contract."""
    try:
        return (
            [float(x) for x in df["close"]],
            [float(x) for x in df["high"]],
            [float(x) for x in df["low"]],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "gp data must be a DataFrame with 'close', 'high' and 'low' columns"
        ) from exc


def _metrics(close: list[float], high: list[float], low: list[float],
             genes: list, engine: str) -> list[float]:
    """Raw strategy metrics via Julia (or its exact Python twin)."""
    ema_len, atr_len, atr_mult = genes
    if engine == "julia":
        return jb.fitness(close, high, low, ema_len, atr_len, atr_mult)
    return jb.fitness_python(close, high, low, ema_len, atr_len, atr_mult)


def _evaluate(close: list[float], high: list[float], low: list[float],
              genes: list, cfg: GPConfig, engine: str) -> tuple[float, dict]:
    """Score one typed parameter vector.

    Returns ``(score, metrics)`` where ``metrics`` always carries the raw
    Julia metrics plus a ``gated`` flag; gated candidates get ``REJECT_SCORE``.
    """
    net, win_rate, pf, max_dd, trades = _metrics(close, high, low, genes, engine)

    if trades < cfg.min_trades or max_dd > cfg.max_drawdown_pct:
        return REJECT_SCORE, {
            "genes": list(genes), "net_pnl": net, "win_rate": win_rate,
            "profit_factor": pf, "max_dd_pct": max_dd, "trades": int(trades),
            "gated": True, "score": REJECT_SCORE,
        }

    bloat = sum(
        abs(g - mid) for g, mid in zip(genes, cfg._midpoints)
    )
    dd_penalty = cfg.dd_penalty_weight * max(0.0, max_dd - cfg.dd_free_band)
    score = float(net) - dd_penalty - cfg.parsimony_weight * bloat

    return score, {
        "genes": list(genes), "net_pnl": net, "win_rate": win_rate,
        "profit_factor": pf, "max_dd_pct": max_dd, "trades": int(trades),
        "gated": False, "score": score,
    }


def evaluate_genes(df, genes: list, cfg: GPConfig | None = None,
                   engine: str | None = None) -> tuple[float, dict]:
    """Score a single typed parameter vector (public, used by tests/UI).

    ``engine`` may be forced to ``"python"`` to bypass Julia.
    """
    cfg = cfg or GPConfig()
    engine = engine or ("julia" if jb.available() else "python")
    close, high, low = _as_arrays(df)
    return _evaluate(close, high, low, list(genes), cfg, engine)


def run_gp(df, cfg: GPConfig | None = None) -> dict:
    """Run the parametric GP over the strategy family and return the champion.

    ``df`` is a DataFrame with ``close``/``high``/``low`` columns (the M1 data
    layer contract). Returns a JSON-safe summary with the hall of fame, the
    evaluation tally and a compact logbook.

    Note: DEAP draws from the process-global ``random`` module, so ``run_gp``
    re-seeds it for reproducibility — any caller relying on ambient randomness
    afterwards will observe the seeded stream.
    """
    cfg = cfg or GPConfig()
    engine = "julia" if jb.available() else "python"
    close, high, low = _as_arrays(df)

    random.seed(cfg.seed)

    from deap import base, creator, tools  # local import: optional dependency
    from deap.algorithms import varAnd

    if not hasattr(creator, "FitnessGP"):
        creator.create("FitnessGP", base.Fitness, weights=(1.0,))
        creator.create("IndividualGP", list, fitness=creator.FitnessGP)

    toolbox = base.Toolbox()
    for i, gene in enumerate(GENE_DEFS):

        def _attr(i=i, gene=gene):
            if gene["kind"] == "int":
                return random.randint(gene["low"], gene["high"])
            return round(random.uniform(gene["low"], gene["high"]), 3)

        toolbox.register(f"attr_{i}", _attr)

    attrs = [toolbox.__getattribute__(f"attr_{i}") for i in range(len(GENE_DEFS))]

    def _individual():
        return creator.IndividualGP([a() for a in attrs])

    toolbox.register("individual", _individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def _fix(individual) -> list:
        """Restore per-gene bounds and int/float kinds after variation."""
        for i, gene in enumerate(GENE_DEFS):
            individual[i] = min(max(individual[i], gene["low"]), gene["high"])
            if gene["kind"] == "int":
                individual[i] = int(round(individual[i]))
        return individual

    def _mate(a, b):
        tools.cxBlend(a, b, alpha=0.5)
        return _fix(a), _fix(b)

    def _mutate(ind):
        tools.mutGaussian(
            ind, mu=(0.0, 0.0, 0.0),
            sigma=(8.0, 4.0, 0.6), indpb=0.4,
        )
        return (_fix(ind),)

    toolbox.register("mate", _mate)
    toolbox.register("mutate", _mutate)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_size)

    tally = {"evals": 0, "gated": 0, "accepted": 0}

    def _deap_evaluate(ind):
        score, metrics = _evaluate(close, high, low, ind, cfg, engine)
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

    # Initial population
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    hof.update(pop)
    logbook.record(gen=0, nevals=len(pop), **stats.compile(pop))

    # Generations — every offspring is re-scored each generation (no fitness
    # caching), so the evaluation tally stays exactly population × (gen + 1).
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
        "engine": engine,
        "genes": [dict(g) for g in GENE_DEFS],
        "best": (
            {
                "genes": list(best),
                "score": float(best.fitness.values[0]),
                "metrics": dict(best.metrics),
            }
            if best is not None
            else None
        ),
        "hall_of_fame": [
            {"genes": list(ind), "score": float(ind.fitness.values[0])}
            for ind in hof
        ],
        "evals": tally["evals"],
        "gated": tally["gated"],
        "accepted": tally["accepted"],
        "logbook": list(logbook),
    }
