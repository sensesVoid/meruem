"""Tests for the M6 Indicator Interpretation Agent + sandboxed Backtest Guard."""

import json

import pytest

from qimera.app.gp_indicators import INDICATOR_IDS
from qimera.app.interp_agent import (
    BoundedWorkflow, InterpAgent, WorkflowHalted, _fallback_signal_code,
    _keyword_resolve, run_backtest_guard,
)


# ---------------------------------------------------------------------------
# Bounded workflow engine
# ---------------------------------------------------------------------------


def test_bounded_workflow_audits_transitions():
    calls = []

    def n1(state):
        calls.append("n1")
        return "n2"

    def n2(state):
        calls.append("n2")
        return None

    wf = BoundedWorkflow().add_node("n1", n1).add_node("n2", n2)
    state = wf.run({}, start="n1")
    assert calls == ["n1", "n2"]
    assert [a["node"] for a in state["audit"]] == ["n1", "n2"]
    assert all("ts" in a for a in state["audit"])


def test_bounded_workflow_step_budget_exceeded():
    def loop(state):
        return "loop"

    wf = BoundedWorkflow(max_steps=3).add_node("loop", loop)
    with pytest.raises(WorkflowHalted):
        wf.run({}, start="loop")


def test_bounded_workflow_unknown_node():
    wf = BoundedWorkflow()
    with pytest.raises(WorkflowHalted):
        wf.run({}, start="missing")


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------


def test_keyword_resolve_maps_families_and_direction():
    assert _keyword_resolve("sell the liquidity sweep")["indicator"] == "liquidity"
    assert _keyword_resolve("sell the liquidity sweep")["direction"] == "short"
    assert _keyword_resolve("Buy the FVG retest")["indicator"] == "fvg"
    assert _keyword_resolve("Buy the FVG retest")["direction"] == "long"
    assert _keyword_resolve("session VWAP mean reversion")["indicator"] == "vwap"
    assert _keyword_resolve("CVD divergence")["indicator"] == "cvd"
    assert _keyword_resolve("order block rejection")["indicator"] == "order_blocks"


# ---------------------------------------------------------------------------
# Agent: LLM path
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Returns a parse JSON then a valid code body (LLM path)."""

    def __init__(self, code=None):
        self.calls = []
        self.code = code

    def __call__(self, prompt):
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return '{"indicator": "cvd", "direction": "long", "rationale": "test"}'
        return self.code or (
            "import numpy as np\n"
            "import pandas as pd\n"
            "from qimera.app.gp_indicators import signal_for\n"
            "def generate_signal(df):\n"
            "    return signal_for('cvd', df, [2.0, 20, 1])\n"
        )


def _fake_search(query, limit=3):
    return [{
        "title": f"Definition of {query}",
        "url": "https://example.com/def",
        "snippet": "A trading concept definition.",
    }]


def test_interp_agent_llm_path(tmp_path):
    agent = InterpAgent(llm=_FakeLLM(), search=_fake_search,
                        provenance_root=tmp_path)
    record = agent.interpret("buy on CVD divergence")
    assert record["indicator"] == "cvd"
    assert record["direction"] == "long"
    assert record["llm_used"] is True
    assert record["code_source"] == "llm"
    assert record["guard"]["ok"] is True
    assert record["guard"]["length"] == 400
    assert record["provenance"][0]["url"] == "https://example.com/def"
    assert "retrieved_at" in record["provenance"][0]
    assert record["schema"]["id"] == "cvd"
    assert len(record["audit"]) >= 6
    assert record["run_id"]


def test_interp_agent_provenance_is_logged(tmp_path):
    agent = InterpAgent(llm=_FakeLLM(), search=_fake_search,
                        provenance_root=tmp_path)
    agent.interpret("CVD momentum")
    lines = (tmp_path / "interpretations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["indicator"] == "cvd"
    assert entry["provenance"][0]["url"] == "https://example.com/def"


# ---------------------------------------------------------------------------
# Agent: offline fallback (no LLM, no network)
# ---------------------------------------------------------------------------


def test_interp_agent_offline_fallback(tmp_path):
    agent = InterpAgent(llm=lambda prompt: "", search=lambda q, limit=3: [],
                        provenance_root=tmp_path)
    record = agent.interpret("buy the liquidity sweep")
    assert record["indicator"] == "liquidity"
    assert record["direction"] == "long"
    assert record["llm_used"] is False
    assert record["code_source"] == "template"
    assert record["provenance"] == []
    assert record["guard"]["ok"] is True


def test_interp_agent_llm_bad_code_retries_template(tmp_path):
    bad_llm = _FakeLLM(code="def generate_signal(df):\n    raise ValueError('boom')")
    agent = InterpAgent(llm=bad_llm, search=_fake_search,
                        provenance_root=tmp_path)
    record = agent.interpret("FVG retest")
    # First LLM attempt fails the guard, template retry must pass.
    assert record["code_source"] == "template_retry"
    assert record["guard"]["ok"] is True


def test_interp_agent_rejects_unknown_llm_indicator(tmp_path):
    def llm(prompt):
        return '{"indicator": "bogus", "direction": "either"}'

    agent = InterpAgent(llm=llm, search=_fake_search, provenance_root=tmp_path)
    record = agent.interpret("VWAP reversion")
    assert record["indicator"] in INDICATOR_IDS  # fell back to keyword
    assert record["indicator"] == "vwap"


# ---------------------------------------------------------------------------
# Sandboxed Backtest Guard
# ---------------------------------------------------------------------------


def test_guard_accepts_valid_template_code():
    code = _fallback_signal_code("cvd", [2.0, 20, 1])
    v = run_backtest_guard(code, timeout=5.0)
    assert v["ok"] is True
    assert v["timed_out"] is False
    assert v["length"] == 400


def test_guard_rejects_forbidden_import():
    code = "import os\nos.system('calc')\ndef generate_signal(df): return df['close']"
    v = run_backtest_guard(code)
    assert v["ok"] is False
    assert "forbidden import" in v["error"]


def test_guard_rejects_import_aliasing_and_multi():
    # Regex scans miss these; the AST allowlist must catch them.
    code = "import os, sys\ndef generate_signal(df): return df['close']"
    v = run_backtest_guard(code)
    assert v["ok"] is False
    assert "forbidden import" in v["error"]

    code2 = "import socket as s\ndef generate_signal(df): return df['close']"
    v2 = run_backtest_guard(code2)
    assert v2["ok"] is False
    assert "forbidden import" in v2["error"]


def test_guard_rejects_forbidden_builtin():
    # ``eval`` is not in the restricted builtins, so the exec raises NameError
    # inside the sandbox (interpreter-level, not a string scan).
    code = "def generate_signal(df):\n    return eval('df[\"close\"]')\n"
    v = run_backtest_guard(code)
    assert v["ok"] is False
    assert v["timed_out"] is False
    assert v["error"] != ""


def test_guard_blocks_builtins_dict_import_trick():
    # Classic scan bypass: ``__builtins__['__import__']``. The restricted
    # builtins dict means __import__ is the guarded wrapper (or absent), so
    # importing 'socket' must fail inside the sandbox.
    code = (
        "def generate_signal(df):\n"
        "    b = __builtins__\n"
        "    imp = b['__import__'] if isinstance(b, dict) else b.__dict__['__import__']\n"
        "    imp('socket')\n"
        "    return df['close']\n"
    )
    v = run_backtest_guard(code)
    assert v["ok"] is False
    assert "socket" in v["error"] or "ImportError" in v["error"]


def test_guard_rejects_empty_code():
    v = run_backtest_guard("   ")
    assert v["ok"] is False
    assert "empty" in v["error"]


def test_guard_kills_runaway_loop():
    code = "def generate_signal(df):\n    while True:\n        pass\n"
    v = run_backtest_guard(code, timeout=0.5)
    assert v["ok"] is False
    assert v["timed_out"] is True


def test_guard_reports_crash():
    code = "def generate_signal(df):\n    raise ValueError('boom')\n"
    v = run_backtest_guard(code, timeout=5.0)
    assert v["ok"] is False
    assert v["timed_out"] is False
    assert "boom" in v["error"]
