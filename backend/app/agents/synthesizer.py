"""Synthesizer Agent (RAG-grounded strategy synthesis).

Phase of strategy tab revamp (#8.2): turns a trader's TRADING CONCEPT (plus optional
notes) into a concrete, machine-actionable ENTRY CRITERIA that is STRICTLY GROUNDED
in the strategy concepts already ingested in the vector store (RAG), plus the trader's
own concept. It never invents rules out of thin air and never copies a single RAG
document whole-cloth: every rule must trace to either the trader's concept or a
retrieved, previously-ingested document. Output still travels through QA before it is
persisted / backtested.
"""

# Module-level import (not ``from ... import DEFAULT_REGISTRY``) so the new-tool
# proposal loop can ``importlib.reload`` the registry and this module always
# reads the CURRENT registry at call time (hot-swappable primitives, IR.md).
from sqlmodel import Session

import app.services.ir_primitives as _ir_primitives
from app.agents.ollama_client import parse_json
from app.agents.validator import ValidatorChain
from app.services.ir import IRCompilerEngine
from app.services.rag import RagService

SYNTHESIZER_PROMPT = """\
You are Meruem Synthesizer Agent. A trader gave a TRADING CONCEPT and wants a concrete,
machine-actionable ENTRY CRITERIA. Synthesize it STRICTLY GROUNDED in the ingested
strategy concepts retrieved from the RAG library below, plus the trader's own concept.
You are the QA partner: the entry criteria you produce must pass the QA gate, so follow
QA's requirements exactly.

Strategy name: {name}
Symbol: {symbol}   Timeframe hint: {timeframe}   Asset class: {asset_class}

TRADING CONCEPT (derive the entry basis FROM THIS — do not change the concept):
---BEGIN CONCEPT---
{concept}
---END CONCEPT---

STRUCTURED CONCEPT TEMPLATE SECTIONS (extracted from the trader's document when \
the template was used — authoritative context; use the tools, mechanics and \
levels verbatim):
---BEGIN TEMPLATE SECTIONS---
{template}
---END TEMPLATE SECTIONS---

ANY EXISTING ENTRY NOTES THE TRADER WROTE (may be empty; keep what is usable):
---BEGIN EXISTING NOTES---
{notes}
---END EXISTING NOTES---

GROUNDED CONCEPTS RETRIEVED FROM THE INGESTED STRATEGY LIBRARY (RAG):
---BEGIN RAG CONCEPTS---
{rag}
---END RAG CONCEPTS---

SAVED REFERENCE STRATEGIES THIS SYNTHESIS IS COMPILED FROM (combine their ideas \
into ONE machine-actionable entry; they are the source library — merge, don't prefer \
one over others):
---BEGIN REFERENCES---
{references}
---END REFERENCES---

LEARNING CONTEXT FROM RECENT LOSING TRADES (may be empty; use only to avoid repeating \
the same mistakes — do NOT add rules that contradict the concept, only strengthen it):
---BEGIN LTR LESSONS---
{lessons}
---END LTR LESSONS---

REQUIREMENTS (QA-compliant, hard constraints):
1. Write the ENTRY CRITERIA as numbered markdown rules. It must define ALL of:
   a. EXACT ENTRY condition using OHLCV data only (candles, close, high, low,
      open, volume, or derived indicators: SMA/EMA/RSI/ATR/breakouts/MSS/FVG).
   b. EXACT STOP-LOSS (e.g. "below the last swing low" or "1.5 x ATR below entry").
   c. EXACT TAKE-PROFIT target(s) (e.g. "2 x ATR above entry" or a level).
   d. Opposing/conflicting triggers (e.g. "if BUY and SELL both fire, take no trade").
   e. Position/risk rule if any reference implies one.
2. You are the one who BUILDS the machine-actionable trade from the CONCEPTS. The
   concepts are ideas (e.g. "ICT: liquidity sweep then imbalance"), NOT yet executable —
   translate them into exact candle/price conditions. Support PRICE ACTION mechanics
   (liquidity sweep, double-top/bottom, breaks of structure CHOCH/MSS/BSL, fair-value
   gaps / imbalances, order-void, retests) and ORDERFLOW mechanics (unbalanced range,
   volume / buy-sell imbalance) by expressing them as deterministic OHLCV rules. Say
   exactly what you approximated from OHLCV in "approximations".
3. GROUND EVERY RULE: trace to a RAG concept, a saved reference, or the trader's own
   concept. Cite inline like [RAG:1] / [REF:n] / [CONCEPT]. Do NOT invent indicators or
   thresholds present in none of them.
4. NO lookahead bias: never use future prices, "tomorrow's close", "the next candle",
   "exit when it later hits X".
5. NO placeholders, "X", "TBD", or generic filler. Every rule must be executable.
6. When combining multiple references, this is a SYNTHESIS: merge the best, most
   consistent ideas into a single coherent setup. If references conflict, pick the
   strongest rationale and note the choice. Do NOT gluing concepts side-by-side.
7. Keep it symbol-agnostic and timeframe-agnostic unless a reference states one.
8. Do NOT copy any individual reference / retrieved concept wholesale; SYNTHESIZE the
   best setup by combining grounded ideas.

IR COMPILER LAYER (IR.md): you must ALSO emit an intermediate-representation
blueprint that maps your entry conditions to the ACTIVE MATHEMATICAL PRIMITIVE
REGISTRY below. Only these primitives may be referenced; the IR engine verifies
your blueprint before the strategy can be backtested, and any rule referencing a
nonexistent primitive is auto-rejected into the self-correcting refine loop.

---BEGIN ACTIVE PRIMITIVE REGISTRY (Tool Schema Menu)---
{ir_schema}
---END ACTIVE PRIMITIVE REGISTRY---

Mapping rules:
1. For EACH executable condition/rule above, add one entry to
   "ir_blueprint": {{"rules": [...]}}. Each rule is
   {{"primitive": NAME, "args": {{all REQUIRED arguments}}}}.
2. Use ONLY primitive names in the registry above; never invent a primitive.
3. Provide every "REQUIRED" argument with the declared type. Non-required
   arguments may be omitted (defaults apply).
4. The first rule should be the ENTRY trigger; later rules carry the stop /
   target / risk mechanics (use ATR_STOP, ATR_TARGET, RISK_PCT as appropriate).
5. Keep the blueprint consistent with the markdown rules — same conditions,
   same thresholds. The blueprint is the machine-executable form of the text.

NEW TOOL DETECTION (optional — only when the trader's concept names a tool that
is NOT in the ACTIVE PRIMITIVE REGISTRY above, e.g. "Momentum Reversal", "OBV
Divergence", "Session VWAP"): define it as a machine-checkable primitive
proposal so it can be registered and used natively next time. For each
unregistered tool the concept actually uses, add one entry to "new_primitives"
(max 3): {{"name": "UPPER_SNAKE_NAME", "description": "deterministic OHLCV
contract", "parameters": {{"param": {{"type": "int"|"float"|"str"|"bool",
"default": <value>}}}}, "formula": "the exact math/conditions in words"}}.
- "name" MUST be UPPER_SNAKE_CASE (e.g. OBV_DIVERGENCE), must NOT already
  exist in the registry above, and must be a real tool named in the concept —
  never invent tools.
- Parameters are lowercase snake_case; give EVERY parameter a default so the
  tool stays backward compatible (omit the default only to force an argument).
- "formula" is required: the deterministic math/conditions the tool computes
  (e.g. "POC = price bin with max volume; enter on retest of the value-area
  low with acceptance back inside").
- Do NOT reference an unregistered tool in "ir_blueprint" — approximate it in
  the markdown rules until it is registered.

Return STRICT JSON only:
{{
  "entry_markdown": "the numbered, executable ENTRY CRITERIA (markdown)",
  "approximations": ["what you approximated from OHLCV and how"],
  "grounded_on": ["short note of which RAG concept ids / presence of trader concept rules each part traces to"],
  "summary": "one sentence on the synthesized entry basis",
  "ir_blueprint": {{"rules": [{{"primitive": "...", "args": {{...}}}}]}},
  "new_primitives": [{{"name": "...", "description": "...",
    "parameters": {{...}}, "formula": "..."}}],
  "self_check": {{
    "verdict": "PASS" | "REVISE" | "FAIL",
    "quality_score": 0..100,
    "contamination": true|false,
    "issues": ["one concise issue per rule that doesn't satisfy QA"]
  }}
}}
"""

# Grounding verification (strictly-grounded enforcement, #8.2).
# Tokens that carry no grounding signal (grammar/mechanics). A rule is judged
# only on the CONTENT tokens left after these are removed, so OHLCV mechanics
# ("close above the 20-bar high") never false-flag while "RSI(14) < 30" still
# has to trace to a source that actually mentions RSI.
_GROUNDING_STOPWORDS = {
    "the", "and", "or", "a", "an", "in", "on", "at", "of", "to", "for",
    "with", "when", "then", "if", "than", "from", "into", "after", "before",
    "prior", "next", "first", "last", "only", "also", "both", "its", "it",
    "this", "that", "price", "prices", "candle", "candles", "bar", "bars",
    "close", "closes", "closed", "closing", "open", "opens", "opening",
    "high", "low", "above", "below", "entry", "enter", "enters", "buy",
    "sell", "stop", "sl", "target", "tp", "risk", "level", "levels",
    "long", "short", "trade", "trades", "position", "signal", "setup",
    "rule", "rules", "point", "per", "max", "min", "x", "n", "once",
    "does", "doesn", "not", "no", "yes", "use", "using", "via",
    # citation markers ([CONCEPT] / [REF:n] / [RAG:n]) — not grounding content
    "concept", "rag", "ref", "reference", "references",
    # strategy-concept template headers — structural, carry no grounding signal
    "concepts", "thesis", "definition", "terms", "toolset", "model",
    "invalidation", "name",
}


# --------------------------------------------------------------------------
# Strategy concept template (#8.7): a structured input the trader can fill so
# synthesis reads consistent sections (thesis / terms / toolset / model /
# invalidation / target) instead of freeform prose. Sections are extracted
# verbatim and injected into the prompt as labeled blocks; they also make RAG
# retrieval and the grounding gate more reliable (consistent vocabulary).
# --------------------------------------------------------------------------
_TEMPLATE_SECTIONS: list[tuple[str, str]] = [
    ("concepts", "Concepts / Thesis"),
    ("terms", "Definition of Terms"),
    ("toolset", "Entry Toolset"),
    ("model", "Entry Model"),
    ("invalidation", "Invalidation / Stop"),
    ("target", "Target"),
]

_TEMPLATE_ORDER = [k for k, _ in _TEMPLATE_SECTIONS]
_TEMPLATE_LABELS = {k: h for k, h in _TEMPLATE_SECTIONS}


def _template_sections(markdown: str) -> dict[str, str]:
    """Extract the structured sections when the author used the template.

    Headers may be written as ``Concepts / Thesis:``, ``## Entry Toolset``, or
    ``Entry Model`` — leading markdown markers and trailing colons are ignored.
    Sections that are empty are dropped, so partial templates still work.
    """
    import re

    def _norm(line: str) -> str:
        s = re.sub(r"^[#*\-\s]+", "", line.strip())
        return s.rstrip(":").strip().lower()

    out: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in markdown.splitlines():
        matched = None
        for key, header in _TEMPLATE_SECTIONS:
            if _norm(line) == header.lower():
                matched = key
                break
        if matched:
            if current is not None and buf:
                out[current] = "\n".join(buf).strip()
            current = matched
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None and buf:
        out[current] = "\n".join(buf).strip()
    return {k: v for k, v in out.items() if v.strip()}


def _build_template_block(sections: dict[str, str]) -> str:
    """Render extracted template sections as labeled blocks for the prompt."""
    if not sections:
        return "(no structured template sections)"
    parts = []
    for key in _TEMPLATE_ORDER:
        text = (sections.get(key) or "").strip()
        if text:
            parts.append(f"{_TEMPLATE_LABELS[key]}:\n{text}")
    return "\n\n".join(parts)


def _rule_blocks(markdown: str) -> list[str]:
    """Split numbered markdown rules into individual blocks."""
    import re

    parts = re.split(r"(?m)^\s*\d+[.)]\s+", markdown)
    rules = [p.strip() for p in parts[1:] if p.strip()]
    if not rules:
        rules = [markdown.strip()]
    return rules


def _reference_has_content(references: list[dict] | None) -> bool:
    """True if any saved reference actually carries a concept to synthesize from."""
    return any((r.get("concept_markdown") or "").strip() for r in (references or []))


def _references_as_concept(references: list[dict] | None) -> str:
    """Collapse the saved reference library into a trader-context block so the
    synthesis can be driven by references alone (no typed concept required)."""
    blocks: list[str] = []
    for r in (references or []):
        text = (r.get("concept_markdown") or "").strip()
        if not text:
            continue
        blocks.append(f"### {r.get('name') or 'Reference Strategy'}\n{text}")
    return "\n\n".join(blocks)


def _build_references_block(references: list[dict] | None,
                            budget: int = 8000) -> str:
    """Join saved reference concepts so EVERY saved reference is present.

    The total budget is split fairly across the references (each capped to its
    share), so a growing library never silently drops an older strategy out of
    the synthesis prompt — previously references beyond the budget were dropped
    WHOLE, which collapsed a large library down to a single (most recent) entry.

    Oversized references are condensed into their structured template sections
    (thesis / terms / toolset / model / invalidation / target) rather than raw
    head characters, so the essential idea of every reference survives.
    """
    entries = [
        ((r.get("name") or "Strategy").strip(),
         (r.get("concept_markdown") or "").strip())
        for r in references or []
    ]
    entries = [e for e in entries if e[1]]
    if not entries:
        return "(no saved references)"
    per_ref = max(600, budget // len(entries))
    blocks: list[str] = []
    for name, text in entries:
        if len(text) <= per_ref:
            blocks.append(f"- {name}:\n  {text}")
            continue
        sections = _template_sections(text)
        if sections:
            head = f"- {name} (condensed):"
            parts = []
            for key in _TEMPLATE_ORDER:
                section_text = (sections.get(key) or "").strip()
                if section_text:
                    parts.append(f"  {_TEMPLATE_LABELS[key]}: {section_text}")
            condensed = "\n".join(parts)
            if len(condensed) > per_ref:
                condensed = condensed[:per_ref].rstrip() + "\n  …(truncated)"
            blocks.append(f"{head}\n{condensed}")
        else:
            blocks.append(
                f"- {name}:\n  {text[:per_ref].rstrip()}\n  …(truncated)")
    return "\n---\n".join(blocks)


def _build_rag_block(rag_docs: list[str], budget: int = 6000) -> str:
    """Join retrieved RAG docs so EVERY retrieved doc is present.

    Each doc is capped to a fair share of the total budget instead of dropping
    whole overflow docs, so a dense retrieval is never truncated down to its
    first hit.
    """
    docs = [d.strip() for d in rag_docs or [] if d and d.strip()]
    if not docs:
        return "(no grounded concepts retrieved)"
    per_doc = max(500, budget // len(docs))
    return "\n---\n".join(
        d if len(d) <= per_doc else d[:per_doc] + "\n…(truncated)"
        for d in docs)


_HEURISTIC_TEMPLATE = """\
1. Enter LONG when price closes above the {n}-bar high of the prior {n} bars \
(breakout) AND the trend indicator confirms (price above its {slow}-SMA).
2. Stop-loss: {n} x ATR(14) below the entry price.
3. Take-profit: 2 x ATR(14) above the entry price (or at the last swing high).
4. Enter SHORT when price closes below the {n}-bar low of the prior {n} bars \
AND price is below its {slow}-SMA.
5. If both LONG and SHORT conditions fire on the same bar, take no trade.
6. Risk 1% of equity per trade; no more than one position open at a time.
"""

# Deterministic QA keywords (merged from the retired StrategyQA agent).
_HEURISTIC_KEYWORDS = [
    "entry", "enter", "buy", "sell", "stop", "sl", "target", "tp",
    "risk", "rsi", "sma", "ema", "atr", "breakout", "trend", "support",
    "resistance", "volume", "candle", "close", "open",
]

_CONTAMINATION_KEYWORDS = [
    "trust me", "generated by", "as an ai", "sample strategy",
    "example only", "placeholder", "todo", "insert your",
]

_IR_REPAIR_PROMPT = """\
You are Meruem Synthesizer self-correction loop. Your previous ir_blueprint was \
rejected by the IR compiler. Fix ONLY the rejected rules below, keep everything \
else identical (entry_markdown, approximations, grounded_on, summary, self_check \
stay unchanged).

ACTIVE PRIMITIVE REGISTRY (Tool Schema Menu):
{ir_schema}

PREVIOUS OUTPUT:
{previous}

IR COMPILER DEFECTS (what to fix — replace hallucinated primitives with registered \
ones and add every missing REQUIRED argument with correct type):
{defects}

Return STRICT JSON only with the same keys, and the corrected ir_blueprint:
{{
  "entry_markdown": "...",
  "approximations": [...],
  "grounded_on": [...],
  "summary": "...",
  "ir_blueprint": {{"rules": [{{"primitive": "...", "args": {{...}}}}]}},
  "self_check": {{"verdict": "PASS" | "REVISE" | "FAIL", "quality_score": 0..100}}
}}
"""

_GROUNDING_REPAIR_PROMPT = """\
You are Meruem Synthesizer grounding self-correction. Your previous ENTRY CRITERIA \
contained rules that do NOT trace back to any ingested source. A rule is grounded \
only when its key terms appear in the TRADING CONCEPT, the SAVED REFERENCES, or the \
RAG concepts below. The machine-actionable OHLCV translation is yours to build, but \
the IDEA, the vocabulary, and the levels must come from the sources — never invent \
indicators, levels, or thresholds present in none of them.

SOURCES (TRADING CONCEPT + SAVED REFERENCES + RAG INGESTED CONCEPTS):
{sources}

UNGROUNDED RULES TO RE-DERIVE STRICTLY FROM THE SOURCES (or drop them if no source \
supports them):
{rules}

Return STRICT JSON only with the same keys as before (entry_markdown, \
approximations, grounded_on, summary, ir_blueprint, self_check), with \
entry_markdown containing the FIXED numbered rules. Keep the already-grounded \
rules identical.
"""

REVIEW_PROMPT = """\
You are Meruem Strategy QA reviewer. A trader submits a TRADING CONCEPT and a \
synthesized ENTRY CRITERIA. Verify the ENTRY CRITERIA is machine-actionable and \
RAG-safe.

TRADING CONCEPT (context only — never reject for being conceptual):
--- {name} / concept ---
{concept}
---

ENTRY CRITERIA (this MUST be machine-actionable — it is backtested and ingested):
--- entry criteria ---
{entry}
---

Judge the ENTRY CRITERIA as a SYSTEMATIC rule. Flag:
- RAG contamination: generic, duplicated, vague, self-referential, or placeholder text.
- Lookahead bias: future prices, "exit when it hits X later", "tomorrow's close",
  "the next candle".
- Loose wording with no concrete entry, stop-loss, target, or risk condition.
- Contradictory / inconsistent conditions (e.g. BUY and SELL on the same trigger).

When acceptable, rewrite it into clean, executable numbered markdown rules while
preserving the author's core idea. NEVER change the trading concept.

Return STRICT JSON only:
{{
  "verdict": "PASS" | "REVISE" | "FAIL",
  "quality_score": 0..100,
  "contamination": true|false,
  "issues": ["one concise issue per entry"],
  "rewritten_markdown": "the cleaned executable ENTRY CRITERIA (or empty if FAIL)",
  "summary": "one short sentence explaining the verdict"
}}
"""


class SynthesizerAgent:
    """Synthesizes a QA-compliant, RAG-grounded entry criteria from a concept."""

    def __init__(self, session: Session | None = None):
        self.session = session
        self.chain = ValidatorChain(agent_id="synthesizer")
        self.ir_compiler = IRCompilerEngine(_ir_primitives.DEFAULT_REGISTRY)

    def _retrieve(self, concept: str, asset_class: str = "",
                  top_k: int = 6) -> list[str]:
        """Pull grounded, ingested concepts from RAG for synthesis.

        Uses :meth:`RagService.search_concepts` — the REFERENCE library (the
        user's ingested concepts) PLUS approved strategies — instead of the
        scanner's ``search_active`` (approved rules only), so the synthesis
        actually grounds on the ingested trading concepts.
        """
        try:
            if self.session is None:
                return []
            rag = RagService(self.session)
            found = rag.search_concepts(
                query=concept, top_k=top_k,
                asset_class=asset_class or None,
            )
            return [d for d in found if isinstance(d, str) and d.strip()]
        except Exception:  # noqa: BLE001 - RAG is best-effort; grounding is a bonus
            return []

    def _grounding_sources(self, concept: str, rag_docs: list[str],
                           references: list[dict] | None) -> list[str]:
        """The full corpus the synthesis must be grounded on: the trader's own
        concept, every saved reference concept, and the RAG-retrieved concepts."""
        sources: list[str] = []
        if concept and concept.strip():
            sources.append(concept.strip())
        for r in (references or []):
            text = (r.get("concept_markdown") or "").strip()
            if text:
                sources.append(text)
        for d in rag_docs or []:
            if isinstance(d, str) and d.strip():
                sources.append(d.strip())
        return sources

    @staticmethod
    def _grounding_check(entry: str, sources: list[str]) -> dict:
        """Deterministic per-rule grounding score (strictly-grounded gate).

        Each numbered rule must share content tokens with the ingested sources.
        ``score`` is the fraction of rules that trace to a source; ``verdict``
        is GROUNDED (>= 0.75) / PARTIAL (>= 0.4) / UNGROUNDED. Rules with only
        mechanical tokens (stopwords) are counted grounded (vacuous), so pure
        execution scaffolding never false-flags.
        """
        import re

        if not sources:
            return {
                "score": 0.0,
                "verdict": "UNGROUNDED",
                "ungrounded_rules": [],
                "source_count": 0,
            }
        source_tokens: set[str] = set()
        for src in sources:
            source_tokens |= {t for t in re.findall(r"[a-z][a-z0-9]{2,}", src.lower())}
        ungrounded: list[str] = []
        grounded = 0
        total = 0
        for rule in _rule_blocks(entry):
            tokens = {t for t in re.findall(r"[a-z][a-z0-9]{2,}", rule.lower())}
            content = tokens - _GROUNDING_STOPWORDS
            total += 1
            if not content:
                grounded += 1  # vacuous/mechanical rule — nothing to verify
                continue
            overlap = content & source_tokens
            coverage = len(overlap) / max(len(content), 1)
            if coverage >= 0.2 or len(overlap) >= 2:
                grounded += 1
            else:
                ungrounded.append(rule[:220])
        score = grounded / max(total, 1)
        verdict = (
            "GROUNDED" if score >= 0.75 else
            "PARTIAL" if score >= 0.4 else "UNGROUNDED"
        )
        return {
            "score": round(score, 2),
            "verdict": verdict,
            "ungrounded_rules": ungrounded,
            "source_count": len(sources),
        }

    def _grounding_gate(self, data: dict, entry: str, concept: str,
                        rag_docs: list[str], references: list[dict] | None,
                        grounding: dict, provider: str) -> tuple[str, dict, dict]:
        """Bounded self-correction for PARTIAL / UNGROUNDED output.

        Feeds the ungrounded rules + the full source texts back so the agent
        re-derives ONLY those rules strictly from the ingested sources (mirrors
        the IR self-correction loop). The repaired entry is adopted when its
        grounding score is no worse; the original data (incl. the ir_blueprint)
        is preserved if the repair omits it.
        """
        if provider == "heuristic":
            return entry, data, grounding
        if grounding.get("verdict") == "GROUNDED":
            return entry, data, grounding
        if not grounding.get("ungrounded_rules"):
            return entry, data, grounding
        import json as _json

        sources = self._grounding_sources(concept, rag_docs, references)
        try:
            prompt = _GROUNDING_REPAIR_PROMPT.format(
                sources=_json.dumps(sources, default=str)[:8000],
                rules="\n".join(grounding.get("ungrounded_rules") or []),
            )
            raw, _prov = self.chain.generate(prompt)
            repaired = parse_json(raw)
            repaired_entry = repaired.get("entry_markdown")
            if isinstance(repaired_entry, list):
                repaired_entry = "\n".join(str(i) for i in repaired_entry).strip()
            elif isinstance(repaired_entry, dict):
                repaired_entry = "\n".join(
                    f"{k}: {v}" for k, v in repaired_entry.items()).strip()
            else:
                repaired_entry = str(repaired_entry or "").strip()
            if not repaired_entry:
                return entry, data, grounding
            if not repaired.get("ir_blueprint") and data.get("ir_blueprint"):
                repaired["ir_blueprint"] = data.get("ir_blueprint")
            for key in ("approximations", "grounded_on", "summary", "self_check",
                        "new_primitives"):
                if key not in repaired:
                    repaired[key] = data.get(key)
            new_grounding = self._grounding_check(repaired_entry, sources)
            if new_grounding["score"] >= grounding.get("score", 0.0):
                return repaired_entry, repaired, new_grounding
            return entry, data, grounding
        except Exception:  # noqa: BLE001 - grounding repair is best-effort
            return entry, data, grounding

    @staticmethod
    def _extract_new_primitives(data: dict) -> tuple[list[dict], list[str]]:
        """Deterministic vetting of the synthesizer's new-tool proposals.

        The LLM may propose tools (from the concept's toolset) that have no
        registered primitive; only proposals that pass the validation gate
        (identifier, types, defaults, registry collision) survive. Returns
        ``(valid_proposals, rejection_reasons)`` so the UI can explain why a
        proposed tool was dropped. LLM output never reaches the codebase
        unvetted.
        """
        raw = data.get("new_primitives")
        if not isinstance(raw, list) or not raw:
            return [], []
        from app.services.primitive_proposals import validate_batch

        valid, errors = validate_batch(
            raw, registry=_ir_primitives.DEFAULT_REGISTRY)
        return valid, errors

    def _ltr_lessons(self) -> str:
        """Learning context for synthesis from reviewed losing trades (LTR).

        Reads the ``ltr_records`` reference DB (``LTRRecord`` snapshots written by
        the LTR agent) and returns a compact lessons block — error + correction —
        so newly synthesized strategies avoid the same mistakes. Best-effort;
        returns empty when there is nothing or no session.
        """
        try:
            from sqlmodel import select

            from app.models.entities import LTRRecord

            if self.session is None:
                return ""
            rows = self.session.exec(
                select(LTRRecord).order_by(LTRRecord.reviewed_at.desc()).limit(6)
            ).all()
            if not rows:
                return ""
            lines = []
            for r in rows:
                reason = (r.loss_reason or "").strip()
                fix = (r.correction or "").strip()
                review = (r.review or "").strip()
                if reason or review:
                    line = f"- {r.symbol} {r.direction} (PnL {r.pnl}): {reason}"
                    if fix:
                        line += f" | Correction: {fix}"
                    if review:
                        line += f" | Lesson: {review}"
                    lines.append(line)
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    def _ir_schema(self) -> str:
        """The Tool Schema Menu injected into the prompt (IRCompiler Layer)."""
        return _ir_primitives.DEFAULT_REGISTRY.generate_agent_prompt_schema()

    def _ir_offline(self) -> dict:
        """Deterministic IR result for the offline/heuristic fallback path."""
        blueprint = {
            "rules": [
                {"primitive": "BREAKOUT",
                 "args": {"n": 20, "direction": "long"}},
                {"primitive": "SMA_CROSS",
                 "args": {"fast": 10, "slow": 50, "direction": "long"}},
                {"primitive": "ATR_STOP",
                 "args": {"period": 14, "multiplier": 1.0}},
                {"primitive": "ATR_TARGET",
                 "args": {"period": 14, "multiplier": 2.0}},
                {"primitive": "RISK_PCT", "args": {"value": 0.01}},
            ]
        }
        result = IRCompilerEngine(_ir_primitives.DEFAULT_REGISTRY).compile_dict(blueprint)
        result["source"] = "heuristic"
        result["blueprint"] = blueprint
        return result

    def _ir_gate(self, data: dict, provider: str, entry: str,
                 max_self_correct: int = 2) -> tuple[str, str, dict]:
        """IR compiler guardrail + closed self-correction loop (IR.md).

        Verifies the agent's ``ir_blueprint`` against the active primitive
        registry. A FAIL verdict triggers a bounded re-prompt where the defects
        are fed back so the agent corrects only its syntax. Returns the final
        (entry_markdown, provider, ir) after repair (or the persisted defects if
        the last attempt still fails).
        """
        import json as _json

        blueprint = data.get("ir_blueprint")
        if not blueprint:
            # No blueprint emitted (e.g. agent didn't follow the schema): keep the
            # original provider and entry — the deterministic QA gate still decides
            # the text — but record that the IR layer was not engaged.
            return entry, provider, {
                "status": "SUCCESS",
                "compiled_blueprint": {"rules": []},
                "note": "No ir_blueprint emitted; IR layer skipped.",
            }

        blob = blueprint if isinstance(blueprint, str) else \
            (blueprint if isinstance(blueprint, (dict, list)) else None)
        result = (self.ir_compiler.compile_and_validate(blob)
                  if isinstance(blob, str) else
                  self.ir_compiler.compile_dict(blob))
        provider_used = provider
        attempts = 0
        while result.get("status") != "SUCCESS" and attempts < max_self_correct:
            attempts += 1
            # -- SELF-CORRECT: bounce the defect block back to the synthesizer --
            try:
                repair_prompt = _IR_REPAIR_PROMPT.format(
                    ir_schema=self._ir_schema(),
                    previous=_json.dumps(data, default=str)[:6000],
                    defects=_json.dumps(result.get("errors") or [],
                                        default=str),
                )
                raw2, prov2 = self.chain.generate(repair_prompt)
                data2 = parse_json(raw2)
                described = data2.get("ir_blueprint")
                if not described:
                    break
                blob2 = described
                result = (self.ir_compiler.compile_and_validate(blob2)
                          if isinstance(blob2, str) else
                          self.ir_compiler.compile_dict(blob2))
                provider_used = prov2
                if result.get("status") == "SUCCESS":
                    # Keep the new-tool proposals (the repair prompt mirrors the
                    # original keys and may omit the optional new_primitives).
                    if "new_primitives" not in data2 and data.get("new_primitives"):
                        data2["new_primitives"] = data.get("new_primitives")
                    data = data2  # repaired blueprint + markdown both adopted
            except Exception:  # noqa: BLE001
                break

        repaired_entry = (data.get("entry_markdown") or entry).strip()
        ir = dict(result)
        ir["provider"] = provider_used
        ir["blueprint_attempts"] = attempts + 1
        return repaired_entry, provider_used, ir

    def synthesize(self, name: str, concept: str, notes: str = "",
                   symbol: str = "EURUSD", timeframe: str = "1h",
                   asset_class: str = "", references: list[dict] | None = None) -> dict:
        """Synthesize a grounded entry criteria from a trading concept + RAG.

        ``references`` (optional) are the saved reference strategies this synthesis
        is COMPILED from (each dict: ``name`` + ``concept_markdown``). They are
        merged into the prompt so multiple saved concepts can be combined into one
        entry. ``session`` (optional) enables RAG retrieval to ground the synthesis.

        The result includes an ``ir`` block: the IRCompilerEngine verdict on the
        synthesized ``ir_blueprint`` (SUCCESS / VALIDATION_FAILED / CRITICAL_ERROR)
        with the validated rules and any defects, per the IR.md layer. A failed
        blueprint triggers the closed self-correction loop (bounded re-prompts);
        a failing strategy is still returned with its defects so the pipeline can
        route it back into the refine loop instead of backtesting uncallable code.
        """
        if not concept or not concept.strip():
            # No trader-typed concept: with a saved reference library (or RAG)
            # the synthesis is still driven by those references — only bail out
            # when there is truly nothing to derive an entry from.
            if not _reference_has_content(references):
                return {
                    "entry_markdown": "",
                    "approximations": [],
                    "grounded_on": [],
                    "summary": "No trading concept provided — nothing to synthesize.",
                    "provider": "heuristic",
                    "grounded": False,
                    "grounding": {
                        "score": 0.0,
                        "verdict": "UNGROUNDED",
                        "ungrounded_rules": [],
                        "source_count": 0,
                        "reason": "No trading concept provided.",
                    },
                    "new_primitives": [],
                    "new_primitive_errors": [],
                    "ir": self._ir_offline(),
                }
            # Reuse the saved references as the authoritative source when no concept
            # was typed, so grounding and the RAG query still have real content.
            concept = _references_as_concept(references)
        rag_docs = self._retrieve(concept, asset_class=asset_class)
        rag_block = _build_rag_block(rag_docs)
        references_block = _build_references_block(references)
        try:
            prompt = SYNTHESIZER_PROMPT.format(
                name=(name or "Strategy")[:120], symbol=symbol,
                timeframe=timeframe, asset_class=asset_class or "unknown",
                concept=concept[:6000], notes=(notes or "")[:2000],
                rag=rag_block,
                references=references_block,
                template=_build_template_block(_template_sections(concept)),
                lessons=self._ltr_lessons()[:4000],
                ir_schema=self._ir_schema(),
            )
            raw, provider = self.chain.generate(prompt)
            data = parse_json(raw)
            raw_entry = data.get("entry_markdown")
            if isinstance(raw_entry, list):
                entry = "\n".join(str(i) for i in raw_entry).strip()
            elif isinstance(raw_entry, dict):
                entry = "\n".join(f"{k}: {v}" for k, v in raw_entry.items()).strip()
            else:
                entry = str(raw_entry or "").strip()

            if not entry:
                raise ValueError("Synthesizer returned no entry criteria.")

            # GROUNDING GATE (strictly-grounded enforcement): verify the entry
            # actually traces to the ingested sources (concept + references +
            # RAG docs). When rules don't, run a bounded self-correction that
            # re-derives ONLY the ungrounded rules from the sources.
            sources = self._grounding_sources(concept, rag_docs, references)
            grounding = self._grounding_check(entry, sources)
            entry, data, grounding = self._grounding_gate(
                data, entry, concept, rag_docs, references, grounding, provider)

            # IR COMPILER LAYER: compile the agent's ir_blueprint. If it fails the
            # guardrail, run the closed self-correction loop (bounded re-prompts
            # feeding the defects back) before accepting the synthesis.
            entry, provider, ir = self._ir_gate(data, provider, entry)
            raw_approx = data.get("approximations") or []
            if isinstance(raw_approx, str):
                approx = [raw_approx]
            elif isinstance(raw_approx, list):
                approx = [str(a) for a in raw_approx]
            else:
                approx = [str(raw_approx)]

            raw_grounded = data.get("grounded_on") or []
            if isinstance(raw_grounded, str):
                grounded = [raw_grounded]
            elif isinstance(raw_grounded, list):
                grounded = [str(g) for g in raw_grounded]
            else:
                grounded = [str(raw_grounded)]

            raw_summary = data.get("summary") or "Entry criteria synthesized."
            if isinstance(raw_summary, list):
                summary = " ".join(str(s) for s in raw_summary)
            else:
                summary = str(raw_summary)

            self_check = self._normalize_self_check(data.get("self_check"))
            new_primitives, new_primitive_errors = self._extract_new_primitives(data)
            return {
                "entry_markdown": entry,
                "approximations": approx,
                "grounded_on": grounded,
                "rag_docs": rag_docs,
                "summary": summary,
                "provider": provider,
                "self_check": self_check,
                "ir": ir,
                "grounded": grounding.get("verdict") == "GROUNDED",
                "grounding": grounding,
                "new_primitives": new_primitives,
                "new_primitive_errors": new_primitive_errors,
                "qa": self._merge_qa(entry, self_check, lenient=True,
                                      grounding=grounding),
            }
        except Exception:  # noqa: BLE001
            fallback = _HEURISTIC_TEMPLATE.format(n=20, slow=50)
            qa_heur = self.review(fallback)
            qa_heur["verdict"] = "REVISE"  # never auto-PASS an ungrounded template
            qa_heur["issues"] = (qa_heur.get("issues") or []) + [
                ("Heuristic template is NOT grounded on the ingested concepts "
                 "(LLM unavailable)."),
            ]
            return {
                "entry_markdown": fallback,
                "approximations": [
                    ("LLM unavailable — used a generic breakout "
                     "template as a starting point.")
                ],
                "grounded_on": [],
                "rag_docs": rag_docs,
                "summary": "Heuristic entry template (LLM offline).",
                "provider": "heuristic",
                "grounded": False,
                "grounding": {
                    "score": 0.0,
                    "verdict": "UNGROUNDED",
                    "ungrounded_rules": [],
                    "source_count": len(rag_docs),
                    "reason": (
                        "Heuristic template — LLM unavailable; not grounded "
                        "on the ingested concepts."
                    ),
                },
                "new_primitives": [],
                "new_primitive_errors": [],
                "ir": self._ir_offline(),
                "self_check": {
                    "verdict": "REVISE",
                    "quality_score": 40,
                    "contamination": False,
                    "issues": ["LLM unavailable — heuristic template; verify before using."],
                },
                "qa": qa_heur,
            }

    @staticmethod
    def _normalize_self_check(raw: object) -> dict:
        if isinstance(raw, dict):
            verdict = str(raw.get("verdict") or "REVISE").upper()
            if verdict not in ("PASS", "REVISE", "FAIL"):
                verdict = "REVISE"
            issues = raw.get("issues") or []
            if isinstance(issues, str):
                issues = [issues]
            return {
                "verdict": verdict,
                "quality_score": float(raw.get("quality_score", 0) or 0),
                "contamination": bool(raw.get("contamination", False)),
                "issues": [str(i) for i in issues],
            }
        return {
            "verdict": "REVISE",
            "quality_score": 0.0,
            "contamination": False,
            "issues": ["No self-check returned by the synthesizer."],
        }

    def _merge_qa(self, entry: str, self_check: dict, lenient: bool = False,
                  grounding: dict | None = None) -> dict:
        """Merge the LLM self-check with the deterministic gate.

        The deterministic heuristic is the AUTHORITATIVE hard gate (cheap,
        offline-safe): only it can FAIL (contamination / emptiness). The LLM
        self-check adds detail and may DOWNGRADE a PASS to REVISE (conservative
        second opinion), but never hard-FAILs alone — the same flash-class model
        that wrote the entry must not be able to silently kill it on a whim.
        ``lenient`` relaxes keyword strictness for the pipeline path.
        ``grounding`` (from :meth:`_grounding_check`) downgrades to REVISE when
        the entry is not strictly grounded on the ingested sources.
        """
        heur = self.review(entry, lenient=lenient)
        llm = self_check
        combined_issues = list(heur.get("issues") or [])
        for issue in llm.get("issues") or []:
            if issue not in combined_issues:
                combined_issues.append(issue)
        if heur["verdict"] == "FAIL":
            verdict = "FAIL"
        elif heur["verdict"] == "REVISE":
            verdict = "REVISE"
        else:
            # heuristic PASS — the LLM may only downgrade, never hard-FAIL alone.
            llm_blob = " ".join(llm.get("issues") or []).lower()
            if (llm.get("contamination") or "contamination" in llm_blob
                    or "lookahead" in llm_blob):
                verdict = "REVISE"
            elif llm["verdict"] == "PASS":
                verdict = "PASS"
            else:
                verdict = "REVISE"
        if grounding and grounding.get("verdict") == "UNGROUNDED":
            verdict = "REVISE"  # strict grounding: never auto-PASS ungrounded output
            issue = ("Entry is not grounded on the ingested concepts — rules do "
                     "not trace to the concept/references/RAG sources.")
            if issue not in combined_issues:
                combined_issues.append(issue)
        return {
            "verdict": verdict,
            "quality_score": min(100.0, max(
                heur.get("quality_score", 0.0), llm.get("quality_score", 0.0))),
            "contamination": bool(heur.get("contamination", False)
                                  or llm.get("contamination", False)),
            "issues": combined_issues,
            "provider": "merged",
        }

    def review(self, markdown: str, lenient: bool = False) -> dict:
        """Deterministic QA gate (proofread + anti-contamination), no LLM.

        This is the merge of the old StrategyQA heuristic: a cheap, offline-safe
        hard gate that must pass before anything is backtested / ingested.
        ``lenient`` relaxes the "needs the literal words entry/stop/target" scoring
        (used when reviewing a machine-actionable trade the synthesizer already
        built), so a price-action/orderflow entry is gated on contamination /
        emptiness rather than on wording.
        """
        import re

        lower = markdown.lower()
        score = 0
        issues: list[str] = []

        contamination = any(k in lower for k in _CONTAMINATION_KEYWORDS)
        if contamination:
            issues.append("Detected placeholder / AI-generated marker text (RAG contamination).")

        if lenient:
            # Gate the machine-actionable entry on STRUCTURE, not wording. The
            # synthesizer already BUILT the trade, so price-action/orderflow entries
            # are accepted if they are substantial and rule-shaped; only contamination
            # and emptiness hard-fail.
            if contamination:
                score = 10
            else:
                length = len(markdown.strip())
                if length < 60:
                    issues.append("Entry criteria is too short to be actionable.")
                    score = 20
                else:
                    score = 68
            numbered_rules = len(re.findall(r"^\s*\d+[.)]", markdown, flags=re.MULTILINE))
            score += min(12, numbered_rules * 6)
            score = max(0, min(100, score))
            if contamination:
                verdict = "FAIL"
            elif score >= 55:
                verdict = "PASS"
            elif score >= 30:
                verdict = "REVISE"
            else:
                verdict = "FAIL"
            return {
                "verdict": verdict,
                "quality_score": score,
                "contamination": contamination,
                "issues": issues,
                "rewritten_markdown": "",
                "provider": "heuristic",
            }

        present = [k for k in _HEURISTIC_KEYWORDS if k in lower]
        score += min(40, len(present) * 4)

        if len(markdown.strip()) < 80:
            issues.append("Strategy is very short; likely too vague to be actionable.")
            score -= 15
        if "\n" not in markdown.strip() and len(markdown.strip()) < 200:
            issues.append("Single-line strategy; consider structured, numbered rules.")
            score -= 5

        numbered_rules = len(re.findall(r"^\s*\d+[.)]", markdown, flags=re.MULTILINE))
        if numbered_rules >= 3:
            score += 15
        elif numbered_rules >= 1:
            score += 5

        if contamination:
            score -= 40

        has_entry = any(k in lower for k in ("entry", "enter", "buy", "sell"))
        has_stop = any(k in lower for k in ("stop", "sl", "risk"))
        has_target = any(k in lower for k in ("target", "tp", "take profit", "profit"))
        if has_entry and has_stop and has_target:
            score += 15
        else:
            if not has_entry:
                issues.append("No concrete entry condition found.")
                score -= 10
            if not has_stop:
                issues.append("No stop-loss / risk management rule found.")
                score -= 25
            if not has_target:
                issues.append("No take-profit / target rule found.")
                score -= 10

        score = max(0, min(100, score))
        if contamination:
            verdict = "FAIL"
        elif score >= 55:
            verdict = "PASS"
        elif score >= 30:
            verdict = "REVISE"
        else:
            verdict = "FAIL"

        return {
            "verdict": verdict,
            "quality_score": score,
            "contamination": contamination,
            "issues": issues,
            "rewritten_markdown": "",
            "provider": "heuristic",
        }

    @staticmethod
    def _review_concept_only(name: str, concept: str) -> dict:
        return {
            "verdict": "REVISE",
            "quality_score": 40,
            "contamination": False,
            "issues": [("No entry criteria provided. Add concrete entry/stop/target "
                       "rules based on this concept so it can be backtested.")],
            "rewritten_markdown": "",
            "summary": "Trading concept accepted; add an entry criteria to backtest.",
            "provider": "heuristic",
        }

    def review_entry(self, name: str, concept: str, markdown: str = "") -> dict:
        """Full QA entry point (merged): LLM self-check when online, else the
        deterministic gate. Replaces ``StrategyQA.review``.
        """
        entry = markdown
        if not entry and not concept:
            return {
                "verdict": "FAIL",
                "quality_score": 0,
                "contamination": True,
                "issues": ["Entry criteria is empty."],
                "rewritten_markdown": "",
                "summary": "Empty entry criteria.",
                "provider": "heuristic",
            }
        if not entry:
            return self._review_concept_only(name, concept=concept)
        if not concept:
            concept = ""
        heur = self.review(entry)
        if heur["verdict"] == "FAIL" or len(entry) < 80:
            # Deterministic gate already rejects; don't burn an LLM call.
            return {**heur, "rewritten_markdown": heur.get("rewritten_markdown", "")}
        try:
            from app.agents.ollama_client import parse_json

            prompt = REVIEW_PROMPT.format(
                name=(name or "Strategy")[:120], concept=(concept or "")[:6000],
                entry=entry[:6000],
            )
            raw, provider = self.chain.generate(prompt)
            data = parse_json(raw)
            llm = self._normalize_self_check({
                "verdict": data.get("verdict"),
                "quality_score": data.get("quality_score"),
                "contamination": data.get("contamination"),
                "issues": data.get("issues"),
            })
            merged = self._merge_qa(entry, llm)
            merged["rewritten_markdown"] = str(data.get("rewritten_markdown") or "")
            merged["summary"] = str(data.get("summary") or "Reviewed.")
            merged["provider"] = provider
            return merged
        except Exception:  # noqa: BLE001
            return {**heur, "rewritten_markdown": heur.get("rewritten_markdown", "")}