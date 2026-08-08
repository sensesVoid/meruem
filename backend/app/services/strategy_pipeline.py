"""Automatic strategy submission pipeline.

Synthesizes (RAG-grounded) -> merged QA gate -> backtest/refine loop -> save in a
background thread so the UI can show live progress. The strategy is only finalized
(its best spec and meaningful status) once the whole QA+optimization gate has
completed successfully and a best setup is found.

Pipeline stages (shown to the user via the job store):
  queued -> synth -> qa -> optimize -> finalize -> done / error
"""

import json
import threading
import uuid

from sqlmodel import Session, select

from app.core.db import engine
from app.models.entities import BacktestRun, Strategy
from app.services.backtest_engine import MIN_OOS_TRADES, BacktestEngine
from app.services.rag import RagService

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()

# Asset-agnostic default: if no symbol is supplied the strategy is validated
# against a representative basket so its edge is not tied to one market.
DEFAULT_SYMBOLS = ["EURUSD", "XAUUSD", "BTCUSDT"]


def _parse_timeframe_hint(markdown: str) -> str | None:
    """Pick a timeframe the strategy mentions in its rules (e.g. "1h", "15m", "4h").
    First explicit token wins; the pipeline still validates on that tf + the basket."""
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
    themselves backtestable; synthesis compiles their ideas into one entry. The
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


def _metrics_green(gate: dict, tolerance: float = 0.35) -> bool:
    """All metrics green on the OUT-OF-SAMPLE (test) window, with an overfit guard.

    The real evidence of a generalizable strategy is the held-out test window
    performing to standard on its own. Requiring the in-sample train window to also
    pass the same bar would wrongly reject good strategies (the model may simply be
    more decisive on unseen data). Overfitting is instead detected as a gap where
    train (in-sample) is far stronger than test (OOS) — a model that excels in-sample
    but collapses out-of-sample is curve-fit.
    """
    if gate["n_trades_test"] < MIN_OOS_TRADES:
        return False
    if gate["win_rate_test"] < 50.0:
        return False
    if gate["sharpe_test"] < 1.0:
        return False
    # Overfit guard: OOS must hold up relative to in-sample, never collapse.
    if gate["sharpe_train"] > 0 and (
        gate["sharpe_test"] < gate["sharpe_train"] * (1 - tolerance)
        and gate["sharpe_test"] < 1.0
    ):
        return False
    return True


def _metrics_overfit_guard(best: dict) -> dict:
    """Human-facing verdict on the best refinement. ALL OUT-OF-SAMPLE metrics must be
    green before a strategy qualifies; high in-sample but collapsing OOS marks
    overfitting and is rejected."""
    train = best.get("train") or {}
    test = best.get("test") or {}
    n_trades_test = test.get("n_trades", 0)
    return {
        "n_trades_train": train.get("n_trades", 0),
        "n_trades_test": n_trades_test,
        "win_rate_train": train.get("win_rate", 0.0),
        "win_rate_test": test.get("win_rate", 0.0),
        "sharpe_train": train.get("sharpe", 0.0),
        "sharpe_test": test.get("sharpe", 0.0),
        "max_drawdown_train": train.get("max_drawdown", 0.0),
        "max_drawdown_test": test.get("max_drawdown", 0.0),
        "profit_factor_train": train.get("profit_factor", 0.0),
        "profit_factor_test": test.get("profit_factor", 0.0),
    }


def _recommendations(gate: dict | None = None, green: bool | None = None,
                     scores: dict | None = None, error: str = "") -> list[dict]:
    """Actionable next steps when a synthesis backtest/refine does not pass.

    Deterministic (offline-safe) — derives recommendations from the actual
    failure mode (too few trades, weak win rate, low Sharpe, overfit, sandbox
    rejection) so the user knows what to do next (add more references, look for
    more entry examples, simplify, etc.).
    """
    recs: list[dict] = []
    if green:
        return []  # approved — no recommendations needed
    n = wr = sh = 0
    tr_sh = 0.0
    has_scores = bool(gate or (scores and scores.get("iterations")))
    if gate:
        n = int(gate.get("n_trades_test") or 0)
        wr = float(gate.get("win_rate_test") or 0)
        sh = float(gate.get("sharpe_test") or 0)
        tr_sh = float(gate.get("sharpe_train") or 0)
    elif scores and scores.get("iterations"):
        iters = scores["iterations"]
        accepted = [i for i in iters if i.get("accepted")]
        last = (accepted[-1] if accepted else iters[-1])
        test = (last or {}).get("test") or {}
        n = int(test.get("n_trades") or 0)
        wr = float(test.get("win_rate") or 0)
        sh = float(test.get("sharpe") or 0)

    if error and "sandbox" in str(error).lower():
        recs.append({
            "type": "code",
            "text": ("The generated strategy code was rejected by the safety "
                      "sandbox. The entry may rely on mechanics that cannot be "
                      "expressed safely — simplify the concept or re-synthesize "
                      "with a clearer, more standard setup."),
        })
    if has_scores:
        if n < MIN_OOS_TRADES:
            recs.append({
                "type": "references",
                "text": (f"Too few trades (OOS {n}, need at least {MIN_OOS_TRADES}) — "
                          "the setup is too selective. Save more reference concepts "
                          "and look for more entry examples so the synthesis merges "
                          "richer, less restrictive rules."),
            })
        if n > 0 and wr < 50:
            recs.append({
                "type": "entries",
                "text": (f"Win rate {wr:.1f}% is below the 50% baseline — the edge is "
                          "weak. Add more entry examples / confluences to the library, "
                          "or a mean-reversion or timing reference to improve entry "
                          "quality."),
            })
        if n > 0 and sh < 1.0:
            recs.append({
                "type": "entries",
                "text": (f"OOS Sharpe {sh:.2f} is below the 1.0 benchmark — the edge "
                          "is inconsistent. Broaden the concept with additional "
                          "references and entry examples before re-running."),
            })
        if gate and tr_sh > 0 and sh < tr_sh * 0.65:
            recs.append({
                "type": "overfit",
                "text": ("The setup overfits in-sample: OOS collapses vs train. "
                          "Simplify the concept — fewer, more robust rules — and add "
                          "more diverse references."),
            })
    if error:
        recs.append({
            "type": "error",
            "text": f"Backtest / refine failed: {error}",
        })
    if not recs:
        recs.append({
            "type": "general",
            "text": ("Metrics did not reach the approval gate. Save more trading "
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
                       error: str = "", gate: dict | None = None,
                       green: bool | None = None) -> dict:
    """Rich post-synthesis diagnostics for the UI: the actual synthesized
    strategy, the backtest scores (incl. the Monte-Carlo benchmark) and
    actionable recommendations, plus the visible thinking trace."""
    synth = {
        "concept": concept,
        "entry_markdown": draft.get("entry_markdown") or "",
        "provider": draft.get("provider"),
        "approximations": draft.get("approximations") or [],
        "grounded_on": draft.get("grounded_on") or [],
        "summary": draft.get("summary") or "",
        "self_check": draft.get("self_check"),
        "qa": draft.get("qa"),
        "grounding": draft.get("grounding"),
        "ir": draft.get("ir"),
        "new_primitives": draft.get("new_primitives") or [],
        "new_primitive_errors": draft.get("new_primitive_errors") or [],
        "rag_docs": (draft.get("rag_docs") or [])[:4],
    }
    scores = None
    if strat_id is not None:
        try:
            with Session(engine) as session:
                rows = session.exec(
                    select(BacktestRun).where(BacktestRun.strategy_id == strat_id)
                    .order_by(BacktestRun.created_at.desc()).limit(8)
                ).all()
            if rows:
                scores = {"iterations": [
                    {
                        "iteration": r.iteration,
                        "accepted": bool(r.accepted),
                        "rationale": r.rationale,
                        "train": (json.loads(r.metrics_json or "{}").get("train") or {}),
                        "test": (json.loads(r.metrics_json or "{}").get("test") or {}),
                    }
                    for r in reversed(rows)
                ]}
        except Exception:  # noqa: BLE001 - diagnostics are best-effort
            scores = None
    return {
        "synth": synth,
        "scores": scores,
        "recommendations": _recommendations(gate, green, scores, error),
        "thinking": _thinking_trace(draft),
    }


def _run_worker(job_id: str, payload: dict) -> None:
    tf_hint = _parse_timeframe_hint(payload.get("body_markdown") or "")
    symbol = payload.get("symbol") or DEFAULT_SYMBOLS[0]
    timeframe = payload.get("timeframe") or tf_hint or "1h"
    bars = int(payload.get("bars") or 6000)
    max_iterations = int(payload.get("max_iterations") or 6)
    min_winrate = float(payload.get("min_winrate") or 55.0)
    min_sharpe = float(payload.get("min_sharpe") or 1.0)
    concept = (payload.get("concept_markdown") or "").strip()

    strat = None
    strat_id: str | None = None
    drafted_entry = False
    draft: dict = {}
    try:
        if not concept:
            raise RuntimeError(
                "A trading concept is required. Describe your idea and the "
                "Synthesizer Agent will build the entry criteria for it.")

        _set(job_id, status="running", stage="synth", progress=0.03,
             message="Synthesizing entry criteria from concept (grounded on RAG)…")

        submitted_entry = (payload.get("body_markdown") or "").strip()
        concept = (concept or submitted_entry).strip()

        # AUTO-SYNTHESIS (single path concept->entry, #8): the Synthesizer Agent
        # (RAG-grounded) ALWAYS derives the executable ENTRY CRITERIA from the
        # trading concept, self-verifies it, and returns the merged QA gate. The
        # standalone QA agent was retired in favor of this deterministic + LLM
        # self-check that rides inside the same call chain.
        with Session(engine) as session:
            from app.agents.synthesizer import SynthesizerAgent

            draft = SynthesizerAgent(session).synthesize(
                payload["name"], concept, notes="",
                symbol=symbol, timeframe=timeframe,
                asset_class=payload.get("asset_class") or "",
                references=_reference_sources(exclude_id=payload.get("strategy_id")),
            )
        qa = draft.get("qa") or {}
        ir = draft.get("ir") or {}
        grounding = draft.get("grounding") or {}
        grounded = bool(draft.get("grounded"))
        # Output that cannot go LIVE: heuristic template (LLM offline) or an
        # entry with ZERO source trace after the grounding self-correction.
        # PARTIAL (>= 0.4) is allowed — most rules ARE grounded — so only these
        # two cases block auto-activation.
        live_blocked = bool(
            draft.get("provider") == "heuristic"
            or grounding.get("verdict") == "UNGROUNDED")
        if draft.get("entry_markdown"):
            drafted_entry = True
            submitted_entry = draft["entry_markdown"]

        # QA-rewritten body is the clean, executable ENTRY CRITERIA we optimize;
        # the concept is preserved as-is for ingestion.
        body = (qa.get("rewritten_markdown") or submitted_entry).strip()

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

            # Only now persist the QA-cleaned body.
            strat.body_markdown = body
            session.add(strat)
            session.commit()

            _set(job_id, stage="optimize", progress=0.1,
                 strategy_id=strat.id,
                 message="AI is interpreting your rules into executable strategy code…")

            from app.agents.strategy_codegen import StrategyCodeGen, heuristic_strategy
            from app.services.code_sandbox import validate_code

            engine_ = BacktestEngine()
            try:
                gen = StrategyCodeGen()
                try:
                    compiled_rules = (ir.get("compiled_blueprint") or {}).get(
                        "rules") or []
                    generated = gen.generate(
                        strat.name, concept, body, symbol, timeframe,
                        ir_blueprint=compiled_rules or None)
                except Exception:  # noqa: BLE001
                    # LLM unavailable — fall back to a safe proxy template so the
                    # pipeline still completes (it is still backtested faithfully).
                    generated = heuristic_strategy()
                code = generated.get("code") or ""
                ok, cerr = validate_code(code)
                if not ok:
                    raise ValueError(f"AI strategy rejected by safety sandbox: {cerr}")
                result = engine_.refine_code_loop(
                    session, strat, symbol, timeframe, code,
                    bars, max_iterations, min_winrate, min_sharpe,
                    confluence=generated.get("confluence") or {},
                    concept=concept,
                )
            except ValueError as exc:
                strat.status = "failed"
                strat.qa_notes = str(exc)
                session.add(strat)
                session.commit()
                _set(job_id, stage="optimize", progress=1.0,
                     diagnostics=_build_diagnostics(
                         draft, concept, strat.id, error=str(exc)))
                raise RuntimeError(str(exc))

            best = result.get("best")
            if best is None:
                strat.status = "failed"
                strat.qa_notes = "No acceptable setup was found after refinement."
                session.add(strat)
                session.commit()
                _set(job_id, stage="optimize", progress=1.0,
                     diagnostics=_build_diagnostics(
                         draft, concept, strat.id,
                         error="No acceptable setup was found after refinement."))
                raise RuntimeError("No acceptable setup was found after refinement.")

            gate = _metrics_overfit_guard(best)
            green = _metrics_green(gate)

            # Apply the best code + confluence spec onto the strategy.
            strat.strategy_code = best.get("code") or code
            strat.params_json = json.dumps(best.get("confluence") or {})
            # STRICT GROUNDING: only grounded (or partially-grounded, post-repair)
            # output may auto-go ACTIVE. The offline heuristic template completes
            # the pipeline (product design) but always lands as a DRAFT, never live.
            if live_blocked:
                strat.status = "draft"
                strat.qa_notes = (
                    "Entry was not grounded on the ingested concepts (e.g. LLM "
                    "unavailable) — review the synthesized entry before approving.")
            else:
                strat.status = "active" if green else "draft"
                strat.qa_notes = ""
            session.add(strat)
            session.commit()
            RagService(session).index_strategy(strat)
            session.refresh(strat)

            # Store green-verdict on the best accepted BacktestRun for approval use.
            accepted_run = session.exec(
                select(BacktestRun).where(
                    BacktestRun.strategy_id == strat.id, BacktestRun.accepted == 1
                ).order_by(BacktestRun.created_at.desc())
            ).first()
            if accepted_run is not None:
                accepted_run.accepted = 1 if green else 0
                session.add(accepted_run)
                session.commit()

            # Capture plain values BEFORE the session closes; reading attributes
            # off a detached instance (expired on commit) raises a refresh error.
            strat_id = strat.id
            result_gate = gate
            result_green = green
            result_spec = best.get("confluence") or {}
            result_had_code = bool(best.get("code") or code)
            result_status = strat.status
            result_live_blocked = live_blocked

        drafted_note = " Entry criteria were auto-drafted by AI." if drafted_entry else ""
        if result_status == "active":
            result_message = "Strategy optimized and active — all metrics green."
        elif result_live_blocked:
            result_message = ("Strategy saved as draft — the synthesized entry was "
                              "not grounded on the ingested concepts; review it "
                              "before approving.")
        else:
            result_message = ("Strategy saved as draft; metrics did not all pass "
                              "the approval gate yet.")
        _set(
            job_id, status="done", stage="done", progress=1.0,
            strategy_id=strat_id,
            message=result_message + drafted_note,
            gate=result_gate,
            green=result_green,
            spec=result_spec,
            faithful=result_had_code,
            drafted_entry=drafted_entry,
            grounded=grounded,
            grounding=grounding,
            ir=ir,
            diagnostics=_build_diagnostics(
                draft, concept, strat_id,
                gate=result_gate, green=result_green),
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
        # Keep the meaningful stage (e.g. "optimize" = synthesis succeeded but
        # backtest/refine failed) so the UI can show where it stopped.
        current = _get(job_id) or {}
        stage = current.get("stage")
        if stage not in ("synth", "qa", "optimize", "finalize"):
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