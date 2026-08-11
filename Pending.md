# Pending Execution — Roadmap & phase status

**Status: current.** The ML-scanner / backtest / GP-evolution / strategy-codegen
stack was **removed** by the raw-LLM refactor (see below). This file tracks the
live roadmap and remaining work.

---

## Refactor (done) — "raw LLM multi-agent workspace"

The old pipeline relied on the ML Scanner (XGBoost on extracted parquet datasets),
a faithful-code backtest engine, a backtest-driven refinement loop, and a GP
evolution layer (`evolution-plan.md`). All of that is gone. Strategy flow is now
single-path and LLM-native:

1. **Synthesis** — `SynthesizerAgent` (RAG-grounded) compiles a saved concept
   (or reference library) into concrete ENTRY CRITERIA plus **proxy parameters**
   (`fast`, `slow`, `rsi_period`, `atr_period`). A merged QA gate (deterministic +
   LLM self-check) rides inside the same call chain and returns the verdict.
2. **QA verdict routing** — FAIL → strategy kept as `failed` for editing;
   UNGROUNDED (non-heuristic) → failed loudly; heuristic/offline → saved as
   `draft`; PASS + grounded → `active`. Approval is a plain status flip.
3. **Scanner** — the raw-LLM `ScannerAgent` reasons over every active strategy's
   rules + proxy params, the reference library, and per-symbol **ML artifact
   context files**, over real computed indicators. No strategy code is generated
   or executed.

Deleted modules: `backtest_engine`, `dataset_service`, `ml_scanner`,
`strategy_codegen`, `ir_codegen`, `code_sandbox`, `gp/`, `xgb_scanner`,
`evolution_db`, `genome`. `IR.md` primitives (FVG/VWAP/volume profile) survive as
the compiler-vocabulary guardrail inside synthesis.

---

## Current status

| Area | Status |
|---|---|
| Strategy pipeline (Synthesis → QA → finalize) | Done |
| Proxy params → scanner indicators | Done |
| Raw-LLM scanner (rules + refs + ml-context) | Done |
| ML artifact context files (`/strategies/ml-artifacts`) | Done |
| IR primitive registry + hot-swap | Done |
| LTR (losing-trade review → synthesis learning context) | Done |
| Broker connector (MT5 / Binance / MEXC), copier, position manager | Done |
| Risk interception (caps, drawdown, news-block) | Done |
| Backend tests (`pytest`) | Green (201 passed) |
| Frontend typecheck + build (`tsc -b`, `npm run build`) | Green |

## Remaining / candidate work

- **Scanner self-evaluation** — a lightweight LLM self-check of each scan's setup
  against its rules before a setup is emitted (heuristic offline fallback).
- **LTR feedback loop into synthesis priors** — reweight the synthesizer's
  grounding on closed-trade outcomes (LTR records are already read-only exposed).
- **Live-forward validation** — track `active` strategy outcomes and surface
  per-strategy drift so stale strategies can be de-activated.
- **Re-introducing search without codegen** (optional) — a pure parameter/primitive
  combinational search over the IR registry is possible in the future; it would
  reuse the existing QA gate as its filter. Not started, not planned.
