"""Tests for the IR Compiler Layer + Dynamic Tool Registry (IR.md, Phase 6).

Covers: registry signature introspection + schema menu generation, the
IRCompilerEngine guardrail (SUCCESS / VALIDATION_FAILED / CRITICAL_ERROR,
fail-forward), the closed self-correction loop in the SynthesizerAgent, and the
pipeline routing of compiled blueprints into the code generator.
"""

import json

from app.services.ir import DynamicToolRegistry, IRCompilerEngine
from app.services.ir_primitives import DEFAULT_REGISTRY, build_default_registry


def test_registry_introspects_signature():
    registry = build_default_registry()
    entry = registry.get("SMA_CROSS")
    assert entry is not None
    params = entry["parameters"]
    assert params["fast"] == {"type": "int", "default": 10}
    assert params["slow"] == {"type": "int", "default": 50}
    assert params["direction"] == {"type": "str", "default": "long"}
    assert entry["description"] and entry["callable"]


def test_registry_requires_arguments():
    registry = build_default_registry()
    entry = registry.get("BREAKOUT")
    params = entry["parameters"]
    assert params["n"]["default"] == "REQUIRED"


def test_schema_menu_is_json_and_lists_primitives():
    menu = DEFAULT_REGISTRY.generate_agent_prompt_schema()
    parsed = json.loads(menu)
    names = {item["primitive"] for item in parsed}
    assert {"SMA_CROSS", "ATR_STOP", "BREAKOUT", "RISK_PCT"} <= names
    item = next(i for i in parsed if i["primitive"] == "BREAKOUT")
    assert "required_arguments" in item
    assert item["required_arguments"]["n"]["type"] == "int"


def test_compile_success():
    result = IRCompilerEngine(DEFAULT_REGISTRY).compile_and_validate(json.dumps({
        "rules": [
            {"primitive": "BREAKOUT", "args": {"n": 20, "direction": "long"}},
            {"primitive": "ATR_STOP", "args": {"period": 14, "multiplier": 1.5}},
        ]
    }))
    assert result["status"] == "SUCCESS"
    assert result["compiled_blueprint"]["rules"][0]["primitive"] == "BREAKOUT"


def test_compile_rejects_hallucinated_primitive():
    result = IRCompilerEngine(DEFAULT_REGISTRY).compile_and_validate(json.dumps({
        "rules": [{"primitive": "TOTALLY_FAKE", "args": {}}]
    }))
    assert result["status"] == "VALIDATION_FAILED"
    assert any("Hallucinated" in e for e in result["errors"])
    assert result["rejected_rules"]


def test_compile_rejects_missing_required_arg():
    result = IRCompilerEngine(DEFAULT_REGISTRY).compile_and_validate(json.dumps({
        "rules": [{"primitive": "BREAKOUT", "args": {"direction": "long"}}]
    }))
    assert result["status"] == "VALIDATION_FAILED"
    assert any("requires argument 'n'" in e for e in result["errors"])


def test_compile_rejects_wrong_arg_type():
    result = IRCompilerEngine(DEFAULT_REGISTRY).compile_and_validate(json.dumps({
        "rules": [{"primitive": "BREAKOUT", "args": {"n": "twenty", "direction": "long"}}]
    }))
    assert result["status"] == "VALIDATION_FAILED"
    assert any("must be int" in e for e in result["errors"])


def test_compile_rejects_bad_json():
    result = IRCompilerEngine(DEFAULT_REGISTRY).compile_and_validate("not json {{{")
    assert result["status"] == "CRITICAL_ERROR"


def test_compile_keeps_valid_rules_among_rejections():
    """Fail-forward: valid rules are retained even when others are rejected."""
    result = IRCompilerEngine(DEFAULT_REGISTRY).compile_and_validate(json.dumps({
        "rules": [
            {"primitive": "BREAKOUT", "args": {"n": 20, "direction": "long"}},
            {"primitive": "NOT_REAL", "args": {}},
        ]
    }))
    assert result["status"] == "VALIDATION_FAILED"
    assert result["rejected_rules"]
    assert result["validated_rules"][0]["primitive"] == "BREAKOUT"


def test_compile_allows_optional_args_defaults():
    """Optional args may be omitted — defaults apply (IR.md: hot-swappable)."""
    result = IRCompilerEngine(DEFAULT_REGISTRY).compile_and_validate(json.dumps({
        "rules": [{"primitive": "ATR_TARGET", "args": {"period": 14}}]
    }))
    assert result["status"] == "SUCCESS"


def test_fix_instruction_present_on_failure():
    result = IRCompilerEngine(DEFAULT_REGISTRY).compile_and_validate(json.dumps({
        "rules": [{"primitive": "GHOST", "args": {}}]
    }))
    assert result["fix_instruction"]


def test_toolset_primitives_registered_with_contracts():
    """The template toolset primitives (FVG filter, VWAP, Volume Profile,
    Bullish Engulfing) are registered with optional, defaulted arguments so the
    IR compiler accepts them and the synthesizer schema menu lists them."""
    registry = build_default_registry()
    for name in ("VWAP_BAND", "VOLUME_PROFILE", "BULLISH_ENGULFING"):
        assert registry.get(name) is not None, f"{name} must be registered"

    fvg = registry.get("FVG_IMBALANCE")
    assert fvg["parameters"]["min_gap_atr"]["type"] == "float"
    assert fvg["parameters"]["min_gap_atr"]["default"] == 0.0
    assert fvg["parameters"]["lookback"]["default"] == 3

    menu = json.loads(DEFAULT_REGISTRY.generate_agent_prompt_schema())
    names = {item["primitive"] for item in menu}
    assert {"VWAP_BAND", "VOLUME_PROFILE", "BULLISH_ENGULFING"} <= names


def test_compile_vwap_and_fvg_filter_blueprint():
    """A blueprint using the new toolset primitives with their arguments
    compiles clean (SUCCESS) — the compiler vocabulary follows the registry."""
    result = IRCompilerEngine(DEFAULT_REGISTRY).compile_dict({
        "rules": [
            {"primitive": "VWAP_BAND",
             "args": {"period": 0, "distance_atr": 0.5, "direction": "long"}},
            {"primitive": "FVG_IMBALANCE",
             "args": {"lookback": 3, "direction": "long", "min_gap_atr": 1.0}},
            {"primitive": "BULLISH_ENGULFING",
             "args": {"lookback": 3, "direction": "long"}},
        ]
    })
    assert result["status"] == "SUCCESS"
    compiled = result["compiled_blueprint"]["rules"]
    assert compiled[1]["args"]["min_gap_atr"] == 1.0


def test_register_requires_primitive_attention():
    """Registration must capture a REQUIRED param correctly."""
    registry = DynamicToolRegistry()

    @registry.register_primitive("MEAN_REVERSION", "test primitive")
    def _mr(stddev: float, lookback: int = 5) -> dict:
        return {"stddev": stddev, "lookback": lookback}

    params = registry.get("MEAN_REVERSION")["parameters"]
    assert params["stddev"]["default"] == "REQUIRED"
    assert params["stddev"]["type"] == "float"
    assert params["lookback"]["default"] == 5


def test_synthesizer_prompt_contains_ir_schema():
    """The synth prompt format includes the IR tool schema menu (Step 1)."""
    from app.agents.synthesizer import SYNTHESIZER_PROMPT
    assert "{ir_schema}" in SYNTHESIZER_PROMPT
    assert "ir_blueprint" in SYNTHESIZER_PROMPT


def test_synthesize_ir_blueprint_compiles_clean(monkeypatch):
    """A clean blueprint lands in the synthesize result with IR SUCCESS."""
    from app.agents.synthesizer import SynthesizerAgent

    def fake_generate(self, prompt):
        assert "ir_schema" in prompt
        return (
            ('{"entry_markdown": "1. Buy on breakout.\\n2. Stop below swing low.\\n'
             '3. Target 2R at liquidity. [CONCEPT]", "approximations": [], '
             '"grounded_on": [], "summary": "ok", "ir_blueprint": '
             '{"rules": [{"primitive": "BREAKOUT", "args": {"n": 20, '
             '"direction": "long"}}, {"primitive": "ATR_STOP", "args": '
             '{"period": 14, "multiplier": 1.5}}]}}'),
            "test",
        )

    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(ValidatorChain, "generate", fake_generate)
    d = SynthesizerAgent().synthesize("IR", "Breakout momentum on pullbacks")
    assert d["ir"]["status"] == "SUCCESS"
    assert d["ir"]["compiled_blueprint"]["rules"][0]["primitive"] == "BREAKOUT"


def test_synthesize_ir_closed_self_correction_loop(monkeypatch):
    """When the agent hallucinates a primitive, the closed self-correction loop
    re-prompts with the defects; a repaired blueprint yields SUCCESS."""
    from app.agents.synthesizer import SynthesizerAgent

    calls = {"n": 0}

    def fake_generate(self, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                json.dumps({
                    "entry_markdown": "1. Buy.",
                    "approximations": [],
                    "grounded_on": [],
                    "summary": "ok",
                    "ir_blueprint": {"rules": [
                        {"primitive": "TICK_FAKE", "args": {"n": 20}}]},
                }),
                "test",
            )
        return (
            json.dumps({
                "entry_markdown": "1. Buy.",
                "approximations": [],
                "grounded_on": [],
                "summary": "ok",
                "ir_blueprint": {"rules": [
                    {"primitive": "BREAKOUT",
                     "args": {"n": 20, "direction": "long"}}]},
            }),
            "test",
        )

    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(ValidatorChain, "generate", fake_generate)
    d = SynthesizerAgent().synthesize("IR", "Breakout.")
    assert calls["n"] == 2, "a second generate (repair) should have run"
    assert d["ir"]["status"] == "SUCCESS"


def test_synthesize_ir_fails_forward_after_repairs(monkeypatch):
    """A defective blueprint that never fixes still returns VALIDATION_FAILED with
    the defect block (fail-forward — no crash, no silent pass)."""
    from app.agents.synthesizer import SynthesizerAgent

    def fake_generate(self, prompt):
        return (
            json.dumps({
                "entry_markdown": "1. Buy.",
                "approximations": [],
                "grounded_on": [],
                "summary": "ok",
                "ir_blueprint": {"rules": [
                    {"primitive": "NOPE_A", "args": {}}]},
            }),
            "test",
        )

    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(ValidatorChain, "generate", fake_generate)
    d = SynthesizerAgent().synthesize("IR", "Breakout.")
    assert d["ir"]["status"] in ("VALIDATION_FAILED", "CRITICAL_ERROR")
    assert d["ir"].get("errors")
    assert d["provider"] == "test"


def test_synthesize_offline_ir_valid(monkeypatch):
    """LLM down: the heuristic fallback IR blueprint is valid by construction."""
    from app.agents.synthesizer import SynthesizerAgent

    def boom(*a, **k):
        raise RuntimeError("all providers down")

    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(ValidatorChain, "generate", boom)
    d = SynthesizerAgent().synthesize("IR", "Breakout momentum.")
    assert d["provider"] == "heuristic"
    assert d["ir"]["status"] == "SUCCESS"