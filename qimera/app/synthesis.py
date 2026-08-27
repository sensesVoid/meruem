"""QiMera Agentic Feature Synthesis (QiMera.md §6.6 / M7).

Two agents cooperate over the M1 Metadata Registry to grow the feature set the
GP / classifier can consume:

- **Evolution Agent** (``EvolutionAgent``) queries ``MetadataRegistry`` and
  proposes *mutations* of registered features: typed parameter perturbations
  (bounds + int/float kind from the registry), normalizer swaps (tanh/sigmoid/
  minmax), and **composite features** combining two base features through a
  vectorized operator (add / subtract / product / ratio). Every proposal is a
  ``FeatureExpr`` with a ``compute(df) -> Series`` implementation.

- **Coder Agent** (``CoderAgent``) writes vectorized NumPy/pandas expressions
  from a ``FeatureExpr`` — a deterministic template generator for the common
  path, plus an LLM refinement hook whose output is **validated by running it
  through the M6 sandboxed Backtest Guard** (AST allowlist + restricted
  builtins + timeout), falling back to the template when the LLM output is
  broken.

- **Candidate Pool** (``FeatureCandidatePool``) collects proposals, scores them
  on a dataset with a caller-supplied metric function, and keeps the top-k —
  the "common validation candidate pool" that M8 (classifier features) and M9
  (champions registry) consume. Outputs are JSON-safe.

The synthesis entry point (``run_synthesis``) chains: seed from registry ->
evolve N proposals -> codegen + guard -> score -> pool.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from qimera.app.feature_library import (
    atr_normalized, clip_bounds, compute_anchored_vwap, detect_fvgs,
    detect_liquidity_sweeps, minmax_scale, order_blocks_proximity,
    price_momentum_norm, rolling_volume_norm, sigmoid_scale, tanh_scale,
)
from qimera.app.feature_library.registry import MetadataRegistry
from qimera.app.interp_agent import run_backtest_guard

# ---------------------------------------------------------------------------
# Base feature dispatch (registry function_name -> df-evaluating callable)
# ---------------------------------------------------------------------------

_BASE_FUNCS: dict[str, Callable] = {
    "atr_normalized": lambda df, period=14: atr_normalized(
        df["high"], df["low"], df["close"], period),
    "rolling_volume_norm": lambda df, period=20: rolling_volume_norm(
        df["volume"], period),
    "price_momentum_norm": lambda df, period=14: price_momentum_norm(
        df["close"], period),
    "detect_fvgs": lambda df, atr_period=14: detect_fvgs(
        df["high"], df["low"], df["close"], atr_period),
    "compute_anchored_vwap": lambda df: compute_anchored_vwap(
        df["high"], df["low"], df["close"], df["volume"],
        pd.Series(False, index=df.index)),  # anchored from bar 0 (no reset)
    "detect_liquidity_sweeps": lambda df, lookback=20: detect_liquidity_sweeps(
        df["high"], df["low"], df["close"], lookback),
    "order_blocks_proximity": lambda df, period=20: order_blocks_proximity(
        df["high"], df["low"], df["close"], df["volume"], period),
}

# Normalizers the coder/evolution can apply on top of a base or composite.
_NORMALIZERS: dict[str, Callable] = {
    "tanh": tanh_scale,
    "sigmoid": sigmoid_scale,
    "minmax": minmax_scale,
    "none": lambda s: clip_bounds(s),
}

# Composite operators: (name, numpy expr as a function of a and b)
_COMPOSITES: dict[str, Callable] = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "product": lambda a, b: a * b,
    "ratio": lambda a, b: a / b.replace(0, np.nan),
}

_DEFAULT_LLM = None  # wired lazily by CoderAgent to avoid import cycles


def _default_llm(prompt: str) -> str:
    """Lazy UniversalLLM call — returns '' on any failure (offline-safe)."""
    try:
        from app.agents.universal_llm import UniversalLLM  # backend package

        text, _ = UniversalLLM(agent_id="qimera_synth").generate(
            prompt, temperature=0.2)
        return text or ""
    except Exception:  # noqa: BLE001 - offline / no backend -> template
        return ""


# ---------------------------------------------------------------------------
# FeatureExpr — one typed, evaluable feature proposal
# ---------------------------------------------------------------------------


@dataclass
class FeatureExpr:
    """A feature proposal: base feature + params, optional composite + normalizer.

    ``compute(df)`` returns a pandas Series normalized to [-1, +1] (the library
    contract), so any proposal can be dropped straight into the M5 signal layer,
    the M8 classifier feature matrix, or the M9 champion record.
    """

    expr_id: str
    base: str                                   # registry function_name
    params: dict[str, Any] = field(default_factory=dict)
    op: str | None = None                       # None | composite op name
    other_base: str | None = None               # second base for composites
    other_params: dict[str, Any] = field(default_factory=dict)
    normalize: str = "tanh"
    source: str = "registry"                    # registry|evolve|llm

    def compute(self, df) -> pd.Series:
        a = self._call_base(self.base, self.params, df)
        if self.op and self.other_base:
            b = self._call_base(self.other_base, self.other_params, df)
            a = _COMPOSITES[self.op](a, b)
        norm = _NORMALIZERS.get(self.normalize, tanh_scale)
        out = norm(a)
        out = pd.Series(np.nan_to_num(out.to_numpy(dtype=float), nan=0.0),
                        index=df.index)
        return out.fillna(0.0)

    @staticmethod
    def _call_base(base: str, params: dict, df) -> pd.Series:
        fn = _BASE_FUNCS.get(base)
        if fn is None:
            raise ValueError(f"unknown base feature {base!r}")
        try:
            return fn(df, **params)
        except TypeError:
            # Feature whose signature has no params (e.g. compute_anchored_vwap).
            return fn(df)

    def summary(self) -> dict:
        return {
            "expr_id": self.expr_id, "base": self.base, "params": self.params,
            "op": self.op, "other_base": self.other_base,
            "other_params": self.other_params, "normalize": self.normalize,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Evolution Agent
# ---------------------------------------------------------------------------


class EvolutionAgent:
    """Queries the Metadata Registry and proposes typed feature mutations."""

    def __init__(self, registry: MetadataRegistry | None = None, seed: int = 42):
        self.registry = registry or MetadataRegistry()
        self._rng = np.random.default_rng(seed)

    def registry_features(self) -> list[dict]:
        return self.registry.list_features()

    def base_expressions(self) -> list[FeatureExpr]:
        """Seed pool: one expression per registry feature, default params."""
        out = []
        for f in self.registry_features():
            fn = f.get("function_name")
            if fn not in _BASE_FUNCS:
                continue
            params = {k: v.get("default") for k, v in f.get("parameters", {}).items()}
            out.append(FeatureExpr(
                expr_id=f"{fn}__default", base=fn,
                params={k: v for k, v in params.items() if v is not None},
                normalize="tanh", source="registry"))
        return out

    def _feature_spec(self, fn: str) -> dict | None:
        return next((f for f in self.registry_features()
                     if f.get("function_name") == fn), None)

    @staticmethod
    def _param_bounds(pdef: dict) -> tuple[float, float]:
        """Typed bounds for a registry param, derived when not declared.

        The M1 registry carries type/default/description per param but not
        low/high; bounds are derived as +-50% around the default (min span 1)
        so mutations always stay sane.
        """
        if pdef.get("low") is not None and pdef.get("high") is not None:
            return float(pdef["low"]), float(pdef["high"])
        default = float(pdef.get("default", 10.0))
        half = max(abs(default) * 0.5, 1.0)
        return default - half, default + half

    def mutate_params(self, expr: FeatureExpr) -> FeatureExpr:
        """Perturb one typed parameter within its registry bounds."""
        spec = self._feature_spec(expr.base)
        params = dict(expr.params)
        pdefs = (spec or {}).get("parameters", {})
        candidates = [k for k in pdefs if k in params]
        if not candidates:
            # Nothing typed to mutate — vary the normalizer instead.
            return self.mutate_normalizer(expr)
        key = str(candidates[int(self._rng.integers(len(candidates)))])
        pdef = pdefs[key]
        low, high = self._param_bounds(pdef)
        span = max((high - low) / 10.0, 1e-6)
        val = float(params.get(key, pdef.get("default", low)))
        val = float(np.clip(val + self._rng.normal(0.0, span), low, high))
        if pdef.get("type") == "integer":
            val = int(round(val))
        params[key] = val
        return FeatureExpr(
            expr_id=f"{expr.base}__mut-{key}", base=expr.base, params=params,
            normalize=expr.normalize, source="evolve")

    def mutate_normalizer(self, expr: FeatureExpr) -> FeatureExpr:
        choices = [n for n in _NORMALIZERS if n != expr.normalize]
        return FeatureExpr(
            expr_id=f"{expr.base}__norm-{expr.normalize}", base=expr.base,
            params=dict(expr.params), normalize=str(choices[int(self._rng.integers(len(choices)))]),
            source="evolve")

    def combine(self, a: FeatureExpr, b: FeatureExpr) -> FeatureExpr:
        """Composite: vectorized operator over two base expressions."""
        op = list(_COMPOSITES)[int(self._rng.integers(len(_COMPOSITES)))]
        return FeatureExpr(
            expr_id=f"{a.base}__{op}__{b.base}", base=a.base,
            params=dict(a.params), op=op, other_base=b.base,
            other_params=dict(b.params), normalize=a.normalize, source="evolve")

    def propose(self, n: int) -> list[FeatureExpr]:
        """N proposals: parameter/normalizer mutations + composites."""
        bases = self.base_expressions()
        if not bases:
            return []
        proposals = []
        for i in range(n):
            a = bases[int(self._rng.integers(len(bases)))]
            roll = float(self._rng.uniform())
            if roll < 0.55:
                proposals.append(self.mutate_params(a))
            elif roll < 0.75:
                proposals.append(self.mutate_normalizer(a))
            else:
                b = bases[int(self._rng.integers(len(bases)))]
                proposals.append(self.combine(a, b))
        return proposals


# ---------------------------------------------------------------------------
# Coder Agent
# ---------------------------------------------------------------------------


class CoderAgent:
    """Writes vectorized NumPy/pandas expressions for a FeatureExpr.

    The deterministic template is the common path; ``refine`` asks an LLM to
    improve it and **validates the reply through the M6 sandboxed Backtest
    Guard** (AST import allowlist + restricted builtins + timeout), falling back
    to the template when the LLM output is broken or offline.
    """

    def __init__(self, llm: Callable | None = None, guard: Callable | None = None):
        self.llm = llm or _default_llm
        self.guard = guard or run_backtest_guard

    def template_code(self, expr: FeatureExpr) -> str:
        """Deterministic vectorized codegen for a FeatureExpr."""
        params_repr = ", ".join(f"{k}={v!r}" for k, v in expr.params.items())
        body = (f"    a = _base('{expr.base}', df, {params_repr})\n")
        if expr.op and expr.other_base:
            oparams = ", ".join(f"{k}={v!r}" for k, v in expr.other_params.items())
            body += (f"    b = _base('{expr.other_base}', df, {oparams})\n"
                     f"    a = _composite('{expr.op}', a, b)\n")
        body += f"    return _normalize('{expr.normalize}', a)\n"
        return (
            "import numpy as np\n"
            "import pandas as pd\n"
            "from qimera.app.feature_library import (\n"
            "    atr_normalized, clip_bounds, compute_anchored_vwap, detect_fvgs,\n"
            "    detect_liquidity_sweeps, minmax_scale, order_blocks_proximity,\n"
            "    price_momentum_norm, rolling_volume_norm, sigmoid_scale, tanh_scale,\n"
            ")\n"
            "def _base(name, df, **params):\n"
            "    if name == 'atr_normalized':\n"
            "        return atr_normalized(df['high'], df['low'], df['close'], **params)\n"
            "    if name == 'rolling_volume_norm':\n"
            "        return rolling_volume_norm(df['volume'], **params)\n"
            "    if name == 'price_momentum_norm':\n"
            "        return price_momentum_norm(df['close'], **params)\n"
            "    if name == 'detect_fvgs':\n"
            "        return detect_fvgs(df['high'], df['low'], df['close'], **params)\n"
            "    if name == 'compute_anchored_vwap':\n"
            "        return compute_anchored_vwap(df['high'], df['low'], df['close'],\n"
            "                                     df['volume'],\n"
            "                                     pd.Series(False, index=df.index))\n"
            "    if name == 'detect_liquidity_sweeps':\n"
            "        return detect_liquidity_sweeps(df['high'], df['low'], df['close'], "
            "**params)\n"
            "    if name == 'order_blocks_proximity':\n"
            "        return order_blocks_proximity(df['high'], df['low'], df['close'],\n"
            "                                      df['volume'], **params)\n"
            "    raise KeyError(name)\n"
            "def _composite(op, a, b):\n"
            "    return {'add': a + b, 'subtract': a - b, 'product': a * b,\n"
            "            'ratio': a / b.replace(0, np.nan)}[op]\n"
            "def _normalize(name, s):\n"
            "    n = {'tanh': tanh_scale, 'sigmoid': sigmoid_scale,\n"
            "         'minmax': minmax_scale, 'none': clip_bounds}[name]\n"
            "    return np.nan_to_num(n(s).to_numpy(dtype=float), nan=0.0)\n"
            "def generate_signal(df):\n"
            + body
            + "\n"
        )

    def refine(self, expr: FeatureExpr, concept: str = "") -> tuple[str, str]:
        """LLM-refined code for ``expr``, validated by the sandboxed guard.

        Returns ``(code, source)`` where source is ``llm`` when the LLM reply
        passed the guard, else ``template``.
        """
        template = self.template_code(expr)
        prompt = (
            "Improve this vectorized pandas signal expression for a trading "
            f"feature. Concept: {concept or expr.summary()}\n"
            "Keep the exact signature `def generate_signal(df)` returning a "
            "numpy array in [-1, 1]. You may adjust thresholds/params but must "
            "stay deterministic and vectorized (no loops over rows). "
            "Return ONLY the function definition, no markdown fences.\n"
            f"Current code:\n{template}")
        reply = self.llm(prompt).strip()
        if not reply or "def generate_signal" not in reply:
            return template, "template"
        verdict = self.guard(reply)
        if not verdict.get("ok"):
            return template, "template"
        return reply, "llm"


# ---------------------------------------------------------------------------
# Candidate Pool + synthesis entry point
# ---------------------------------------------------------------------------


class FeatureCandidatePool:
    """Common validation candidate pool: proposals, scores, top-k selection."""

    def __init__(self, k: int = 10):
        self.k = k
        self._candidates: list[dict] = []

    @property
    def size(self) -> int:
        return len(self._candidates)

    def add(self, expr: FeatureExpr, code: str = "", code_source: str = "",
            score: float | None = None, metrics: dict | None = None) -> dict:
        rec = {
            "candidate_id": uuid.uuid4().hex[:12],
            "expr": expr.summary(),
            "code": code,
            "code_source": code_source,
            "score": score,
            "metrics": metrics or {},
        }
        self._candidates.append(rec)
        return rec

    def evaluate(self, df, metrics_fn: Callable) -> list[dict]:
        """Score every candidate on ``df`` with ``metrics_fn(Series) -> dict``.

        The metrics dict must include a ``score`` key; candidates whose
        evaluation fails are dropped with their error recorded.
        """
        for rec in self._candidates:
            expr = FeatureExpr(**rec["expr"])
            try:
                series = expr.compute(df)
                m = dict(metrics_fn(series))
                rec["score"] = float(m.pop("score", 0.0))
                rec["metrics"] = m
            except Exception as exc:  # noqa: BLE001 - keep pool alive
                rec["score"] = None
                rec["metrics"] = {"error": f"{type(exc).__name__}: {exc}"}
        return self.top()

    def top(self) -> list[dict]:
        ranked = [c for c in self._candidates if c["score"] is not None]
        ranked.sort(key=lambda c: c["score"], reverse=True)
        return ranked[: self.k]


def run_synthesis(df, n_proposals: int = 12, k: int = 10, seed: int = 42,
                  metrics_fn: Callable | None = None,
                  llm: Callable | None = None) -> dict:
    """End-to-end agentic synthesis: registry -> evolve -> codegen+guard -> score.

    ``metrics_fn`` receives a normalized Series and must return a dict with at
    least ``score``; the default scores by signal activity (mean |x|), which
    rewards informative features without any trading assumption.
    """
    evolution = EvolutionAgent(seed=seed)
    coder = CoderAgent(llm=llm)
    pool = FeatureCandidatePool(k=k)

    base = evolution.base_expressions()
    proposals = base + evolution.propose(n_proposals)

    for expr in proposals:
        code, source = coder.refine(expr)
        pool.add(expr, code=code, code_source=source)

    if metrics_fn is None:
        def metrics_fn(series: pd.Series) -> dict:
            return {"score": float(np.abs(series).mean())}

    ranked = pool.evaluate(df, metrics_fn)
    return {
        "n_proposed": len(proposals),
        "n_base": len(base),
        "n_evolved": len(proposals) - len(base),
        "pool_size": pool.size,
        "top": ranked,
    }
