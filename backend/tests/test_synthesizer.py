"""Tests for the Synthesizer Agent (RAG-grounded strategy synthesis, #8.2)."""

from app.agents.synthesizer import SynthesizerAgent

CONCEPT = (
    "Mean-reversion playbook for XAUUSD: trade pullbacks to the value area "
    "against the higher-timeframe trend, enter only after a lower-timeframe "
    "structure shift, keep risk small and target the opposing value boundary."
)


def test_synthesize_requires_concept():
    d = SynthesizerAgent().synthesize("X", "")
    assert d["entry_markdown"] == ""
    assert d["provider"] == "heuristic"
    assert d["grounded_on"] == []


def test_synthesize_heuristic_fallback_has_qa_shape():
    d = SynthesizerAgent().synthesize("X", CONCEPT)
    assert d["entry_markdown"]
    assert "entry" in d["entry_markdown"].lower()
    lower = d["entry_markdown"].lower()
    assert any(k in lower for k in ("stop", "sl"))
    assert any(k in lower for k in ("take-profit", "target", "tp"))
    assert d["provider"] == "heuristic"  # offline path


def test_synthesize_llm_path(monkeypatch):
    def fake_generate(self, prompt):
        return (
            ('{"entry_markdown": "1. Buy when price closes above the 20-bar '
             'high.\\n2. Stop: 1.5 x ATR(14) below entry.\\n3. Target: 2 x ATR above '
             'entry. [CONCEPT]", "approximations": ["test"], "grounded_on": ["lib id 1"], '
             '"summary": "ok"}'),
            "test",
        )

    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(ValidatorChain, "generate", fake_generate)
    d = SynthesizerAgent().synthesize("X", CONCEPT)
    assert d["provider"] == "test"
    assert "Buy when" in d["entry_markdown"]
    assert d["approximations"] == ["test"]
    assert d["grounded_on"] == ["lib id 1"]


def test_synthesize_retrieves_rag_with_session(monkeypatch):
    """RAG retrieval is attempted with a session and lands in the response.
    Retrieval must use search_concepts (the ingested reference + active
    library), NOT the scanner's search_active, so synthesis grounds on the
    user's ingested concepts."""
    calls = {}

    class FakeRag:
        def __init__(self, session):
            calls["session"] = session

        def search_concepts(self, query, top_k=6, asset_class=None):
            calls["query"] = query
            calls["top_k"] = top_k
            calls["asset_class"] = asset_class
            return ["GROUNDED DOC A", "GROUNDED DOC B"]

    monkeypatch.setattr("app.agents.synthesizer.RagService", FakeRag)

    def fake_generate(self, prompt):
        assert "GROUNDED DOC A" in prompt
        return (
            ('{"entry_markdown": "1. Buy on breakout.", "approximations": [], '
             '"grounded_on": ["doc A"], "summary": "ok"}'),
            "test",
        )

    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(ValidatorChain, "generate", fake_generate)
    sess = object()
    d = SynthesizerAgent(sess).synthesize(
        "X", CONCEPT, asset_class="forex", symbol="XAUUSD", timeframe="15m",
    )
    assert calls["session"] is sess
    assert calls["query"] == CONCEPT
    assert calls["top_k"] == 6
    assert calls["asset_class"] == "forex"
    assert d["rag_docs"] == ["GROUNDED DOC A", "GROUNDED DOC B"]
    assert d["grounded_on"] == ["doc A"]


def test_synthesize_grounding_gate_grounded(monkeypatch):
    """Rules whose key terms trace to the concept pass the grounding gate."""
    concept = (
        "Momentum breakout: enter when price closes above the 20-bar high with "
        "RSI(14) above 50; stop below the swing low; target 2R."
    )

    def fake_generate(self, prompt):
        assert concept in prompt
        return (
            ('{"entry_markdown": "1. Buy on the 20-bar high breakout with RSI above 50.\\n'
             '2. Stop below the swing low.\\n3. Target 2R. [CONCEPT]", '
             '"approximations": [], "grounded_on": ["concept"], "summary": "ok"}'),
            "test",
        )

    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(ValidatorChain, "generate", fake_generate)
    d = SynthesizerAgent().synthesize("G", concept)
    assert d["provider"] == "test"
    assert d["grounding"]["verdict"] == "GROUNDED"
    assert d["grounded"] is True
    assert d["grounding"]["ungrounded_rules"] == []


def test_synthesize_ungrounded_rules_trigger_repair(monkeypatch):
    """Rules with zero source overlap trigger the grounding self-correction loop
    (a bounded second LLM call) and never silently pass — the merged QA is REVISE."""
    import json as _json

    concept = "Mean reversion to the VWAP value edge."
    calls = {"n": 0}
    entry = (
        "1. Buy the breakout of the 20-bar high with RSI above 50.\\n"
        "2. Stop below the swing low.\\n3. Target 2R. [CONCEPT]"
    )

    def fake_generate(self, prompt):
        calls["n"] += 1
        return (_json.dumps({
            "entry_markdown": entry,
            "approximations": [],
            "grounded_on": [],
            "summary": "ok",
        }), "test")

    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(ValidatorChain, "generate", fake_generate)
    d = SynthesizerAgent().synthesize("UG", concept)
    assert calls["n"] == 2, "the grounding repair must re-prompt the agent"
    assert d["grounding"]["verdict"] == "UNGROUNDED"
    assert d["grounded"] is False
    assert d["grounding"]["ungrounded_rules"]
    assert d["qa"]["verdict"] == "REVISE", (
        "ungrounded output must never auto-PASS the merged QA gate")


def test_template_sections_extracted_into_prompt(monkeypatch):
    """When the trader uses the concept template, sections are extracted and
    injected into the synthesis prompt as structured, authoritative context."""
    from app.agents.synthesizer import _build_template_block, _template_sections

    concept = """Strategy Name: ICT Sweep

Concepts / Thesis:
  Trade the imbalance left after a liquidity sweep.

Definition of Terms:
  - FVG = Fair value gap
  - CSD = Change in state of delivery

Entry Toolset:
  VWAP, FVG, CHoCH/MSS

Entry Model:
  Sweep then FVG retest entry.

Invalidation / Stop:
  Close below the sweep low.

Target:
  Opposing liquidity pool.
"""
    sections = _template_sections(concept)
    assert sections.get("concepts") and "imbalance" in sections["concepts"].lower()
    assert sections.get("toolset") == "VWAP, FVG, CHoCH/MSS"
    assert "sweep low" in sections.get("invalidation", "").lower()
    assert sections.get("target")

    block = _build_template_block(sections)
    assert "Entry Toolset" in block and "VWAP, FVG, CHoCH/MSS" in block
    assert "Entry Model" in block and "FVG retest" in block

    # Partial template: missing sections are simply dropped, no crash.
    partial = _template_sections("Entry Toolset:\n  RSI only")
    assert partial.get("toolset") == "RSI only"
    assert "target" not in partial

    def fake_generate(self, prompt):
        assert "STRUCTURED CONCEPT TEMPLATE SECTIONS" in prompt
        assert "VWAP, FVG, CHoCH/MSS" in prompt
        return (
            ('{"entry_markdown": "1. Buy the FVG retest after the sweep.\\n'
             '2. Stop below the sweep low.\\n3. Target opposing liquidity. [CONCEPT]", '
             '"approximations": [], "grounded_on": ["concept"], "summary": "ok"}'),
            "test",
        )

    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(ValidatorChain, "generate", fake_generate)
    d = SynthesizerAgent().synthesize("T", concept)
    assert d["provider"] == "test"
    assert "FVG retest" in d["entry_markdown"]


def test_rag_search_concepts_includes_references(session, monkeypatch):
    """search_concepts retrieves the ingested REFERENCE concepts (concept-only
    docs), not just approved bodies — the grounding corpus is actually found."""
    from app.models.entities import Strategy
    from app.services.rag import RagService

    session.add(Strategy(
        id="ref-1", name="ICT",
        concept_markdown="Liquidity sweep then FVG retest.",
        body_markdown="", status="reference",
    ))
    session.add(Strategy(
        id="act-1", name="Trend", concept_markdown="Breakout momentum.",
        body_markdown="1. Buy on breakout. 2. Stop ATR. 3. Target 2R.",
        status="active",
    ))
    session.commit()

    monkeypatch.setattr("app.services.rag._embedder_available", lambda: False)
    docs = RagService(session).search_concepts("sweep liquidity FVG", top_k=5)
    assert any("Liquidity sweep" in d for d in docs), (
        "reference concepts must be retrievable by search_concepts")
    assert all(isinstance(d, str) and d.strip() for d in docs)


def test_synthesize_rag_failure_still_synthesizes(monkeypatch):
    """A RAG outage must not break synthesis (best-effort grounding)."""

    def boom(*a, **k):
        raise RuntimeError("chroma down")

    monkeypatch.setattr("app.agents.synthesizer.RagService", boom)

    def fake_generate(self, prompt):
        return (
            ('{"entry_markdown": "1. Buy.", "approximations": [], "grounded_on": [], '
             '"summary": "ok"}'),
            "test",
        )

    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(ValidatorChain, "generate", fake_generate)
    d = SynthesizerAgent(object()).synthesize("X", CONCEPT)
    assert d["provider"] == "test"
    assert d["rag_docs"] == []
