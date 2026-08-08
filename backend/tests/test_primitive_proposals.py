"""Tests for the new-tool proposal loop (safe registration of IR primitives).

Covers: deterministic validation (name/type/default/registry-collision),
canonical code generation, the pure append-to-file logic, an end-to-end
register + live-reload (with cleanup), the SynthesizerAgent's proposal
extraction, and the API endpoints.
"""

import importlib
import json

from app.services import ir_primitives
from app.services import primitive_proposals as pp


def _good_proposal():
    return {
        "name": "OBV_DIVERGENCE",
        "description": "Divergence between price and on-balance volume.",
        "parameters": {
            "lookback": {"type": "int", "default": 20},
            "direction": {"type": "str", "default": "long"},
        },
        "formula": "OBV rising while price makes a lower low: long on the "
                   "retest of the divergence low.",
    }


def test_validate_accepts_good_proposal():
    ok, err = pp.validate_proposal(_good_proposal(),
                                   registry=ir_primitives.DEFAULT_REGISTRY)
    assert err == ""
    assert ok["name"] == "OBV_DIVERGENCE"
    assert ok["parameters"]["lookback"]["type"] == "int"
    assert ok["parameters"]["direction"]["default"] == "long"


def test_validate_rejects_bad_name():
    for name in ("lowercase", "with space", "X", "HAS-DASH"):
        ok, err = pp.validate_proposal(
            {"name": name, "description": "x",
             "parameters": {"n": {"type": "int", "default": 1}},
             "formula": "x"},
            registry=ir_primitives.DEFAULT_REGISTRY)
        assert ok is None and "name" in err, name


def test_validate_rejects_python_keyword_name():
    """A name whose function form is a Python keyword must never reach the
    code generator (it would emit ``def import(...)`` -> SyntaxError)."""
    for name in ("IMPORT", "CLASS", "RETURN", "LAMBDA"):
        ok, err = pp.validate_proposal(
            {"name": name, "description": "x",
             "parameters": {"n": {"type": "int", "default": 1}},
             "formula": "x"},
            registry=ir_primitives.DEFAULT_REGISTRY)
        assert ok is None and "Python" in err, name


def test_validate_rejects_registered_name():
    ok, err = pp.validate_proposal(
        {"name": "BREAKOUT", "description": "x",
         "parameters": {"n": {"type": "int", "default": 20}}, "formula": "x"},
        registry=ir_primitives.DEFAULT_REGISTRY)
    assert ok is None and "already" in err


def test_validate_rejects_bad_param_type_and_name():
    ok, err = pp.validate_proposal(
        {"name": "PIVOT_REVERSAL", "description": "x",
         "parameters": {"lookback": {"type": "list", "default": 3}},
         "formula": "x"},
        registry=ir_primitives.DEFAULT_REGISTRY)
    assert ok is None and "lookback" in err

    ok, err = pp.validate_proposal(
        {"name": "PIVOT_REVERSAL", "description": "x",
         "parameters": {"Bad Param": {"type": "int", "default": 3}},
         "formula": "x"},
        registry=ir_primitives.DEFAULT_REGISTRY)
    assert ok is None


def test_validate_rejects_type_mismatched_default():
    ok, err = pp.validate_proposal(
        {"name": "ZSCORE_ENTRY", "description": "x",
         "parameters": {"z": {"type": "int", "default": 2.5}}, "formula": "x"},
        registry=ir_primitives.DEFAULT_REGISTRY)
    assert ok is None and "default" in err


def test_validate_defaults_missing_to_type_appropriate():
    ok, _err = pp.validate_proposal(
        {"name": "PIVOT_REVERSAL", "description": "x",
         "parameters": {"lookback": {"type": "int"}}, "formula": "x"},
        registry=ir_primitives.DEFAULT_REGISTRY)
    assert ok is not None and ok["parameters"]["lookback"]["default"] == 0


def test_generate_code_is_canonical_and_executable():
    code = pp.generate_primitive_code(pp.validate_proposal(
        _good_proposal(), registry=ir_primitives.DEFAULT_REGISTRY)[0])
    assert 'register_primitive(\n        "OBV_DIVERGENCE"' in code
    assert "def obv_divergence(lookback: int = 20, direction: str = \"long\")" in code
    assert 'return {"lookback": lookback, "direction": direction}' in code
    # The block is 4-space indented (embedded in build_default_registry) —
    # wrap it in a function to verify it is syntactically valid Python.
    compile("def _f():\n" + code, "<gen>", "exec")


def test_append_to_file_inserts_before_return_registry(tmp_path):
    fake = tmp_path / "ir_primitives.py"
    fake.write_text(
        "def build_default_registry():\n"
        "    registry = None\n"
        "    return registry\n",
        encoding="utf-8")
    prop = pp.validate_proposal(_good_proposal(),
                                registry=ir_primitives.DEFAULT_REGISTRY)[0]
    new_text = pp.append_to_file(fake, [pp.generate_primitive_code(prop)])
    assert 'register_primitive(\n        "OBV_DIVERGENCE"' in new_text
    assert new_text.index("OBV_DIVERGENCE") < new_text.index("    return registry")
    compile(new_text, str(fake), "exec")


def test_register_live_reloads_registry():
    """End-to-end: approving a proposal writes the canonical function to
    ir_primitives.py, reloads the module, and the IR compiler immediately
    accepts blueprints using the new primitive. Cleaned up afterwards."""
    name = "TEST_PROPOSAL_TMP"
    proposal = {
        "name": name,
        "description": "Temporary test primitive for the proposal loop.",
        "parameters": {"lookback": {"type": "int", "default": 10}},
        "formula": "Long on a test condition.",
    }
    assert name not in ir_primitives.DEFAULT_REGISTRY
    try:
        res = pp.register_primitives([proposal])
        assert name in res["registered"]
        assert name in ir_primitives.DEFAULT_REGISTRY

        from app.services.ir import IRCompilerEngine

        r = IRCompilerEngine(ir_primitives.DEFAULT_REGISTRY).compile_dict({
            "rules": [{"primitive": name, "args": {"lookback": 10}}],
        })
        assert r["status"] == "SUCCESS"
    finally:
        try:
            pp.remove_primitives([name])
        finally:
            importlib.reload(ir_primitives)
    assert name not in ir_primitives.DEFAULT_REGISTRY


def test_register_rejects_invalid_batch_without_writing():
    """A batch with zero valid proposals raises and never touches the file."""
    try:
        pp.register_primitives([{"name": "bad-name", "description": "x"}])
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "name" in str(exc)
    # the registry is untouched
    assert "bad-name" not in ir_primitives.DEFAULT_REGISTRY


def test_synthesizer_extracts_and_validates_new_primitives(monkeypatch):
    from app.agents.synthesizer import SynthesizerAgent
    from app.agents.validator import ValidatorChain

    def fake_generate(self, prompt):
        assert "NEW TOOL DETECTION" in prompt
        return (
            json.dumps({
                "entry_markdown": "1. Buy the sweep. 2. Stop below the level. "
                                  "3. Target the opposing liquidity at the sweep. [CONCEPT]",
                "approximations": [], "grounded_on": [], "summary": "ok",
                "ir_blueprint": {"rules": [
                    {"primitive": "BREAKOUT", "args": {"n": 20, "direction": "long"}}]},
                "new_primitives": [
                    _good_proposal(),
                    {"name": "bad-name", "description": "x",
                     "parameters": {}, "formula": "x"},
                    {"name": "BREAKOUT", "description": "dup",
                     "parameters": {}, "formula": "x"},
                ],
            }),
            "test",
        )

    monkeypatch.setattr(ValidatorChain, "generate", fake_generate)
    d = SynthesizerAgent().synthesize(
        "N", "Momentum sweep with OBV divergence confirmation.")
    props = d.get("new_primitives") or []
    names = {p["name"] for p in props}
    assert "OBV_DIVERGENCE" in names
    assert "bad-name" not in names and "BREAKOUT" not in names
    # rejection reasons are surfaced so the UI can explain dropped proposals
    assert d.get("new_primitive_errors"), "rejected proposals must be reported"


def test_diagnostics_surface_new_primitives():
    from app.services.strategy_pipeline import _build_diagnostics

    diag = _build_diagnostics(
        {"entry_markdown": "1. Buy.", "new_primitives": [
            {"name": "OBV_DIVERGENCE", "description": "x",
             "parameters": {"lookback": {"type": "int", "default": 20}},
             "formula": "x"}]},
        concept="c", strat_id=None)
    props = diag["synth"]["new_primitives"]
    assert props and props[0]["name"] == "OBV_DIVERGENCE"
