"""Tests for the M7 agentic feature synthesis (Evolution + Coder agents)."""

import json

import numpy as np
import pandas as pd
import pytest

from qimera.app.feature_library.registry import MetadataRegistry
from qimera.app.synthesis import (
    CoderAgent, EvolutionAgent, FeatureCandidatePool, FeatureExpr,
    run_synthesis,
)


@pytest.fixture
def df():
    np.random.seed(42)
    n = 300
    times = np.arange(1690000000, 1690000000 + n * 900, 900)
    close = 1.09 + np.cumsum(np.random.normal(0, 0.001, n))
    open_p = close + np.random.normal(0, 0.0005, n)
    high = np.maximum(open_p, close) + np.random.uniform(0, 0.001, n)
    low = np.minimum(open_p, close) - np.random.uniform(0, 0.001, n)
    return pd.DataFrame({
        "time": times,
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": np.random.uniform(10, 1000, n),
    })


@pytest.fixture
def registry():
    return MetadataRegistry()


# ---------------------------------------------------------------------------
# Evolution Agent
# ---------------------------------------------------------------------------


def test_base_expressions_cover_registry(registry):
    evo = EvolutionAgent(registry=registry)
    bases = evo.base_expressions()
    assert len(bases) >= 5  # primitives + building blocks that dispatch
    for expr in bases:
        assert expr.source == "registry"
        assert expr.base in _base_names()


def _base_names():
    from qimera.app.synthesis import _BASE_FUNCS
    return set(_BASE_FUNCS)


def test_param_mutation_stays_in_bounds_and_kind(registry):
    evo = EvolutionAgent(registry=registry, seed=3)
    bases = evo.base_expressions()
    for _ in range(40):
        expr = evo.mutate_params(bases[0])
        spec = next(f for f in registry.list_features()
                    if f["function_name"] == expr.base)
        for key, val in expr.params.items():
            pdef = spec["parameters"][key]
            low, high = EvolutionAgent._param_bounds(pdef)
            assert low <= val <= high
            if pdef.get("type") == "integer":
                assert float(val).is_integer()


def test_combine_builds_composite(registry):
    evo = EvolutionAgent(registry=registry, seed=11)
    bases = evo.base_expressions()
    c = evo.combine(bases[0], bases[1])
    assert c.op in ("add", "subtract", "product", "ratio")
    assert c.other_base == bases[1].base


def test_propose_returns_typed_expressions(registry):
    evo = EvolutionAgent(registry=registry, seed=7)
    props = evo.propose(10)
    assert len(props) == 10
    assert all(p.source == "evolve" for p in props)
    assert all(isinstance(p, FeatureExpr) for p in props)


# ---------------------------------------------------------------------------
# FeatureExpr evaluation
# ---------------------------------------------------------------------------


def test_expr_compute_normalized_and_finite(df, registry):
    evo = EvolutionAgent(registry=registry, seed=5)
    for expr in evo.base_expressions() + evo.propose(6):
        s = expr.compute(df)
        assert len(s) == len(df)
        assert s.isna().sum() == 0
        assert s.min() >= -1.0 - 1e-9 and s.max() <= 1.0 + 1e-9


def test_expr_summary_json_serializable(registry):
    evo = EvolutionAgent(registry=registry, seed=2)
    json.dumps(evo.base_expressions()[0].summary())


# ---------------------------------------------------------------------------
# Coder Agent
# ---------------------------------------------------------------------------


def test_template_code_passes_guard_and_matches_compute(df, registry):
    evo = EvolutionAgent(registry=registry, seed=1)
    coder = CoderAgent()
    expr = evo.base_expressions()[0]
    code = coder.template_code(expr)
    assert "def generate_signal" in code
    verdict = coder.guard(code)
    assert verdict["ok"] is True

    # Template semantics must match the direct FeatureExpr evaluation.
    direct = expr.compute(df).to_numpy(dtype=float)
    ns = {}
    exec(code.replace("def generate_signal", "def _gen"), ns)
    from_code = np.asarray(ns["_gen"](df), dtype=float)
    assert np.allclose(direct, from_code, atol=1e-9)


def test_template_code_for_composite(df, registry):
    evo = EvolutionAgent(registry=registry, seed=9)
    bases = evo.base_expressions()
    c = evo.combine(bases[0], bases[1])
    coder = CoderAgent()
    code = coder.template_code(c)
    assert coder.guard(code)["ok"] is True


def test_refine_llm_code_validated_by_guard(registry):
    good_code = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from qimera.app.feature_library import atr_normalized, tanh_scale\n"
        "def generate_signal(df):\n"
        "    a = atr_normalized(df['high'], df['low'], df['close'], period=10)\n"
        "    return np.nan_to_num(tanh_scale(a).to_numpy(dtype=float), nan=0.0)\n"
    )

    class FakeLLM:
        def __call__(self, prompt):
            return good_code

    evo = EvolutionAgent(registry=registry, seed=4)
    expr = evo.base_expressions()[0]
    code, source = CoderAgent(llm=FakeLLM()).refine(expr)
    assert source == "llm"
    assert code == good_code.strip()  # refine strips the LLM reply


def test_refine_bad_llm_code_falls_back_to_template(registry):
    class BadLLM:
        def __call__(self, prompt):
            return "def generate_signal(df):\n    while True:\n        pass\n"

    evo = EvolutionAgent(registry=registry, seed=4)
    expr = evo.base_expressions()[0]
    coder = CoderAgent(llm=BadLLM())
    code, source = coder.refine(expr)
    assert source == "template"
    assert coder.guard(code)["ok"] is True


# ---------------------------------------------------------------------------
# Candidate pool + end-to-end
# ---------------------------------------------------------------------------


def test_pool_scores_ranks_and_keeps_top_k(df, registry):
    evo = EvolutionAgent(registry=registry, seed=8)
    pool = FeatureCandidatePool(k=3)
    for expr in evo.base_expressions()[:5]:
        pool.add(expr, code="", code_source="registry")
    ranked = pool.evaluate(
        df, lambda s: {"score": float(np.abs(s).mean())})
    assert len(ranked) == 3  # top-k kept
    scores = [c["score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(c["candidate_id"] for c in ranked)


def test_pool_survives_evaluation_errors(df, registry):
    evo = EvolutionAgent(registry=registry, seed=6)
    pool = FeatureCandidatePool(k=10)
    pool.add(evo.base_expressions()[0], code="", code_source="registry")
    pool.add(FeatureExpr(expr_id="broken", base="does_not_exist"), code="")
    ranked = pool.evaluate(df, lambda s: {"score": 1.0})
    assert len(ranked) == 1  # broken candidate dropped, pool intact
    assert "error" in pool._candidates[1]["metrics"]


def test_run_synthesis_end_to_end(df, registry):
    res = run_synthesis(df, n_proposals=8, k=5, seed=3)
    assert res["n_base"] >= 5
    assert res["n_evolved"] == 8
    assert res["pool_size"] == res["n_base"] + 8
    assert 1 <= len(res["top"]) <= 5
    assert res["top"][0]["score"] >= res["top"][-1]["score"]
    json.dumps(res)  # JSON-safe
