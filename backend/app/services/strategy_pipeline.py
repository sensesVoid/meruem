"""Automatic strategy submission pipeline.

Synthesizes (RAG-grounded) -> merged QA gate -> finalize in a background thread
so the UI can show live progress. The strategy is only finalized (its clean entry
criteria, proxy parameters and status) once the whole QA gate has completed
successfully. QA verdicts come from the synthesis call chain itself — the
standalone backtest/refine loop was retired in favor of the QA-self-check that
rides inside synthesis.

Pipeline stages (shown to the user via the job store):
  queued -> synth -> qa -> finalize -> done / error
"""

import json
import threading
import uuid

from sqlmodel import Session, select

from app.core.db import engine
from app.models.entities import Strategy
from app.services.rag import RagService

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()

# Asset-agnostic default: if no symbol is supplied the strategy is still
# synthesized fine — the scanner re-reads params per asset at scan time.
# FX-first focus (MT5 / Prop Firm): fallbacks are FX majors.
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]


def _parse_timeframe_hint(markdown: str) -> str | None:
    """Pick a timeframe the strategy mentions in its rules (e.g. "1h", "15m", "4h").
    First explicit token wins."""
    import re

    tokens = re.findall(r"\b(\d{1,3}\s?(?:m|h|d|M)\b)", markdown, flags=re.IGNORECASE)
    for t in tokens:
        t = t.strip().lower()
        if t in {"15m", "30m", "1h", "2h", "4h", "8h", "1d"}:
            return t
        if t in {"1m", "5m"}:
            return t
    return None


def _set(job_id: str, **fields) -> None:
    with _LOCK:
        _JOBS.setdefault(job_id, {})
        _JOBS[job_id].update(fields)


def _get(job_id: str) -> dict | None:
    with _LOCK:
        return dict(_JOBS.get(job_id) or {})


def _reference_sources(exclude_id: str | None = None) -> list[dict]:
    """All saved REFERENCE documents (the source library) to combine into a synthesis.

    References are concept-only strategies (status ``reference``) that are not
    themselves executable; synthesis compiles their ideas into one entry. The
    strategy currently being synthesized is excluded so it stays the primary.
    """
    with Session(engine) as session:
        rows = session.exec(
            select(Strategy).where(Strategy.status == "reference")
        ).all()
    return [
        {"name": s.name, "concept_markdown": s.concept_markdown}
        for s in rows if s.id != exclude_id and (s.concept_markdown or "").strip()
    ]


def _recommendations(qa: dict | None = None, error: str = "") -> list[dict]:
    """Actionable next steps when a synthesis QA gate does not pass.

    Deterministic (offline-safe) — derives recommendations from the actual
    failure mode (QA verdict + reported issues) so the user knows what to do
    next (add more references, write more concrete rules, etc.).
    """
    recs: list[dict] = []
    qa = qa or {}
    if qa.get("verdict") == "PASS":
        return []  # approved — no recommendations needed
    issues = qa.get("issues") or []
    if qa.get("verdict") == "FAIL":
        recs.append({
            "type": "concept",
            "text": ("QA failed — the synthesized entry is not machine-actionable. "
                      "Rewrite the concept with exact entry, stop and take-profit "
                      "conditions and re-synthesize."),
        })
    elif issues:
        recs.append({
            "type": "concept",
            "text": ("QA found issues below — address them in the concept and "
                      "re-synthesize. More concrete, verifiable rules pass the "
                      "gate."),
        })
    for issue in issues[:5]:
        recs.append({
            "type": "qa",
            "text": f"QA issue: {issue}",
        })
    if error:
        recs.append({
            "type": "error",
            "text": f"Synthesis failed: {error}",
        })
    if not recs:
        recs.append({
            "type": "general",
            "text": ("QA did not pass the approval gate. Save more trading "
                      "concepts as references, gather more entry examples, and "
                      "refine the concept before resubmitting."),
        })
    return recs


def _thinking_trace(draft: dict) -> list[dict]:
    """Human-readable trace of the synthesizer's internal steps (RAG retrieval,
    synthesis, grounding check, IR compile, QA) so the user can see HOW the
    entry was built — the 'thinking process' behind the synthesis."""
    trace: list[dict] = []
    rag_docs = draft.get("rag_docs") or []
    trace.append({
        "step": "retrieval",
        "label": "Grounded retrieval (RAG)",
        "detail": (f"{len(rag_docs)} ingested concept(s) retrieved from the library"
                   if rag_docs else "No grounded concepts retrieved (library is empty)."),
        "data": [str(d)[:300] for d in rag_docs[:3]],
    })
    trace.append({
        "step": "synthesis",
        "label": "Entry synthesis",
        "detail": f"Generated via provider '{draft.get('provider') or 'unknown'}'.",
        "data": draft.get("entry_markdown") or "",
    })
    grounding = draft.get("grounding") or {}
    if grounding:
        trace.append({
            "step": "grounding",
            "label": "Grounding check",
            "detail": (f"{grounding.get('verdict')} — "
                       f"{grounding.get('score', 0) * 100:.0f}% of rules trace to the "
                       f"ingested sources ({grounding.get('source_count', 0)} source(s))."),
            "data": (grounding.get("ungrounded_rules") or [])[:5],
        })
    ir = draft.get("ir") or {}
    if ir:
        errors = ir.get("errors") or []
        trace.append({
            "step": "ir",
            "label": "IR compiler guardrail",
            "detail": (f"{ir.get('status')} after {ir.get('blueprint_attempts', 1)} "
                       "attempt(s)."),
            "data": (errors[:3] if errors
                      else (ir.get("compiled_blueprint") or {}).get("rules", [])[:5]),
        })
    qa = draft.get("qa") or {}
    if qa:
        trace.append({
            "step": "qa",
            "label": "QA gate",
            "detail": f"Verdict {qa.get('verdict')} (score {qa.get('quality_score', 0)}).",
            "data": (qa.get("issues") or [])[:5],
        })
    return trace


def _build_diagnostics(draft: dict, concept: str, strat_id: str | None,
                       error: str = "") -> dict:
    """Post-synthesis diagnostics for the UI: the actual synthesized strategy,
    the proxy parameters, actionable recommendations, and the visible thinking
    trace."""
    synth = {
        "concept": concept,
        "entry_markdown": draft.get("entry_markdown") or "",
        "provider": draft.get("provider"),
        "approximations": draft.get("approximations") or [],
        "grounded_on": draft.get("grounded_on") or [],
        "summary": draft.get("summary") or "",
        "params": draft.get("params"),
        "self_check": draft.get("self_check"),
        "qa": draft.get("qa"),
        "grounding": draft.get("grounding"),
        "ir": draft.get("ir"),
        "new_primitives": draft.get("new_primitives") or [],
        "new_primitive_errors": draft.get("new_primitive_errors") or [],
        "rag_docs": (draft.get("rag_docs") or [])[:4],
    }
    return {
        "synth": synth,
        "recommendations": _recommendations(draft.get("qa"), error),
        "thinking": _thinking_trace(draft),
    }


def _run_worker(job_id: str, payload: dict) -> None:
    tf_hint = _parse_timeframe_hint(payload.get("body_markdown") or "")
    symbol = payload.get("symbol") or DEFAULT_SYMBOLS[0]
    timeframe = payload.get("timeframe") or tf_hint or "1h"
    concept = (payload.get("concept_markdown") or "").strip()
    references = _reference_sources(exclude_id=payload.get("strategy_id"))

    strat = None
    strat_id: str | None = None
    drafted_entry = False
    draft: dict = {}
    try:
        if not concept and not references:
            raise RuntimeError(
                "A trading concept is required. Describe your idea (or save "
                "reference documents) and the Synthesizer Agent will build the "
                "entry criteria for it.")

        _set(job_id, status="running", stage="synth", progress=0.03,
             message="Synthesizing entry criteria from concept (grounded on RAG)…")

        submitted_entry = (payload.get("body_markdown") or "").strip()
        concept = (concept or submitted_entry).strip()
        # When no concept was typed but the library has references, keep the saved
        # strategy row readable: store a source summary derived from the references.
        if not concept and references:
            concept = "Synthesized from the saved reference documents:\n" + "\n".join(
                f"- {r.get('name') or 'Reference Strategy'}" for r in references
            )

        # AUTO-SYNTHESIS (single path concept->entry, #8): the Synthesizer Agent
        # (RAG-grounded) ALWAYS derives the executable ENTRY CRITERIA from the
        # trading concept, self-verifies it, and returns the merged QA gate. The
        # QA verdict rides inside the same call chain, so the backtest/refine
        # loop is no longer part of the pipeline.
        with Session(engine) as session:
            from app.agents.synthesizer import SynthesizerAgent

            draft = SynthesizerAgent(session).synthesize(
                payload["name"], concept, notes="",
                symbol=symbol, timeframe=timeframe,
                asset_class=payload.get("asset_class") or "",
                references=references,
            )
        qa = draft.get("qa") or {}
        ir = draft.get("ir") or {}
        grounding = draft.get("grounding") or {}
        grounded = bool(draft.get("grounded"))
        # Output that cannot go LIVE: heuristic template (LLM offline) or an
        # entry with ZERO source trace after the grounding self-correction.
        live_blocked = bool(
            draft.get("provider") == "heuristic"
            or grounding.get("verdict") == "UNGROUNDED")
        if draft.get("entry_markdown"):
            drafted_entry = True
            submitted_entry = draft["entry_markdown"]

        # QA-rewritten body is the clean, executable ENTRY CRITERIA; the concept
        # is preserved as-is for ingestion. Proxy parameters come straight from
        # synthesis — the scanner re-reads them per asset at scan time.
        body = (qa.get("rewritten_markdown") or submitted_entry).strip()
        params = draft.get("params") or {}

        with Session(engine) as session:
            if payload.get("strategy_id"):
                strat = session.get(Strategy, payload["strategy_id"])
                if strat is None:
                    raise RuntimeError("Strategy to update was not found.")
                strat.name = payload["name"]
                strat.concept_markdown = concept
                strat.body_markdown = submitted_entry
                strat.qa_notes = ""
                strat.status = "draft"
                strat.params_json = ""
                session.add(strat)
                session.commit()
                session.refresh(strat)
                strat_id = strat.id
            else:
                strat = Strategy(
                    id=str(uuid.uuid4()),
                    name=payload["name"],
                    concept_markdown=concept,
                    body_markdown=submitted_entry,
                    status="draft",
                )
                session.add(strat)
                session.commit()
                session.refresh(strat)
                strat_id = strat.id

            # A FAIL verdict is a hard stop, but it keeps the (editable) draft row.
            if qa.get("verdict") == "FAIL":
                strat.qa_notes = (
                    qa.get("summary")
                    or "QA failed: strategy is not machine-actionable.")
                strat.status = "failed"
                session.add(strat)
                session.commit()
                session.refresh(strat)
                _set(job_id, status="error", stage="qa", progress=1.0,
                     strategy_id=strat_id, message=strat.qa_notes)
                return

            # STRICT GROUNDING: an LLM-produced entry that traces to NO ingested
            # source (even after the grounding self-correction) is not a strategy
            # — fail loudly so the user refines the concept/references instead of
            # silently persisting hallucinated rules. (Heuristic provider means the
            # LLM was offline; that path completes per the offline design below.)
            if live_blocked and draft.get("provider") != "heuristic":
                rules = "; ".join(grounding.get("ungrounded_rules") or [])[:600]
                strat.status = "failed"
                strat.qa_notes = (
                    "Synthesized entry is not grounded on the ingested concepts."
                    + (f" Ungrounded rules: {rules}" if rules else ""))
                session.add(strat)
                session.commit()
                session.refresh(strat)
                _set(job_id, status="error", stage="synth", progress=1.0,
                     strategy_id=strat.id,
                     message=("Synthesis is not grounded on the ingested "
                              "concepts — refine the concept/references and "
                              "resubmit."))
                return

            # Persist the QA-cleaned body + the proxy parameters that the scanner
            # uses to compute indicators at scan time.
            strat.body_markdown = body
            strat.params_json = json.dumps(params)
            session.add(strat)
            session.commit()

            _set(job_id, stage="finalize", progress=0.9,
                 strategy_id=strat.id,
                 message="Finalizing strategy…")

            # QA PASS + grounded output may auto-go ACTIVE. The offline heuristic
            # template completes the pipeline (product design) but always lands as
            # a DRAFT, never live. Otherwise the strategy stays a DRAFT until the
            # user reviews and approves it.
            if live_blocked:
                strat.status = "draft"
                strat.qa_notes = (
                    "Entry was not grounded on the ingested concepts (e.g. LLM "
                    "unavailable) — review the synthesized entry before approving.")
            elif qa.get("verdict") == "PASS":
                strat.status = "active"
                strat.qa_notes = ""
            else:
                strat.status = "draft"
                strat.qa_notes = qa.get("summary") or "QA did not pass yet."
            session.add(strat)
            session.commit()
            RagService(session).index_strategy(strat)
            session.refresh(strat)

            # Capture plain values BEFORE the session closes; reading attributes
            # off a detached instance (expired on commit) raises a refresh error.
            strat_id = strat.id
            result_params = params
            result_status = strat.status
            result_live_blocked = live_blocked

        drafted_note = " Entry criteria were auto-drafted by AI." if drafted_entry else ""
        if result_status == "active":
            result_message = "Strategy active — QA passed."
        elif result_live_blocked:
            result_message = ("Strategy saved as draft — the synthesized entry was "
                              "not grounded on the ingested concepts; review it "
                              "before approving.")
        else:
            result_message = ("Strategy saved as draft; QA did not pass the gate "
                              "yet — review the synthesized entry before "
                              "approving.")
        _set(
            job_id, status="done", stage="done", progress=1.0,
            strategy_id=strat_id,
            message=result_message + drafted_note,
            params=result_params,
            grounded=grounded,
            grounding=grounding,
            ir=ir,
            diagnostics=_build_diagnostics(draft, concept, strat_id),
        )
    except Exception as exc:  # noqa: BLE001
        # Persist the failure so the user can edit / discard / ask AI to fix it.
        if strat_id is not None:
            try:
                with Session(engine) as session:
                    row = session.get(Strategy, strat_id)
                    if row is not None:
                        row.status = "failed"
                        row.qa_notes = row.qa_notes or str(exc)
                        session.add(row)
                        session.commit()
                        _set(job_id, strategy_id=strat_id)
            except Exception:  # noqa: BLE001
                pass
        # Keep the meaningful stage (e.g. "qa" = synthesis succeeded but QA
        # failed) so the UI can show where it stopped.
        current = _get(job_id) or {}
        stage = current.get("stage")
        if stage not in ("synth", "qa", "finalize"):
            stage = "error"
        extra: dict = {}
        if draft:
            extra["diagnostics"] = _build_diagnostics(
                draft, concept, strat_id, error=str(exc))
        _set(job_id, status="error", stage=stage, message=str(exc), **extra)


def start(payload: dict) -> str:
    job_id = uuid.uuid4().hex
    _set(job_id, status="queued", stage="queued", progress=0.0,
         message="Queued; strategy pipeline starting…")
    threading.Thread(target=_run_worker, args=(job_id, payload), daemon=True).start()
    return job_id


def status(job_id: str) -> dict | None:
    return _get(job_id)
