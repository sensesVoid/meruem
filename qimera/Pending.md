# QiMera Pending Execution & Roadmap

This file tracks the implementation progress, completed milestones, and upcoming tasks for the **QiMera** research/optimization subproject.

---

## Milestone Status Summary

| Milestone | Deliverable | Status |
| :--- | :--- | :--- |
| **M1** | Data Layer & Shared Feature Library v0 | ✅ Completed |
| **M2** | Python/Julia Interop Proof-of-Concept | ✅ Completed |
| **M3** | Single-Asset/Indicator Parametric GP | ✅ Completed |
| **M4** | Walk-Forward Validation Layer & Monte Carlo Gate | ✅ Completed |
| **M5** | Multi-Indicator GP Framework (VWAP, FVG, Liquidity, OB, CVD) | ✅ Completed |
| **M6** | Indicator Interpretation Agent (LLM concept-to-code) | ✅ Completed |
| **M7** | Agentic Feature Synthesis Layer (LLM composite feature proposal) | ✅ Completed |
| **M8** | Unified Rolling-Window XGBoost Classifier | ✅ Completed |
| **M9** | Champions Registry (SQLite) | 📅 Pending |
| **M10** | Export Artifact Packaging & Manifest Schema | 🔄 Partial / Mocked |
| **M11** | Multi-Asset Rollout & Retraining Automation | 📅 Pending |

---

## Detailed Task Checklist

### Phase M1: Data Layer & Feature Library v0 ✅
- [x] Finalize the regime filter interface contract with Meruem's `RegimeAgent` output (`regime_contract.py` — GARCH/HMM one-pass, numerically consistent with `backend/app/services/regime.py`).
- [x] Implement the `Data Layer` to ingest historical OHLCV + tick data and save as Snappy-compressed Parquet files under `qimera/data/` (`data_layer.py`).
- [x] Create the Shared Feature Library skeleton (`qimera/app/feature_library/` — primitives + building blocks).
- [x] Define the `Metadata Registry` schema (`metadata_registry.json` for library primitives, `registry.py`).
- [x] Add normalization functionality ensuring all feature outputs fall within `[-1.0, +1.0]` (`normalization.py`).

> **On-demand data sources (added):** `qimera/app/data_sources.py` fetches OHLCV directly from
> **yfinance** (Yahoo library, internet, no local dependency) or the **local MT5 terminal** (broker
> feed, needs the terminal running, only source for 4H candles) via
> `GET /api/data/ohlcv?symbol=&timeframe=&limit=&start=&end=&source=yfinance|mt5` or
> `ingest_market_data(..., source=...)`. Both return the Meruem candle-dict contract.
>
> **Deep 1m history + lightweight mode (added):** `fetch_mt5_deep` pages the MT5 terminal backwards
> in date windows (`copy_rates_range` triggers deep history sync; position-based paging does NOT
> reach far back) — verified pulling **371,894 one-minute bars (~1 year) of EURUSD in 4.2s**, and
> spot-checked full M1 days back to **2015 (10+ years)** on this broker. yfinance 1m is hard-capped
> to the last **30 days** by Yahoo, so MT5 is the only deep-intraday source. For lightweight use:
> `compact=True` returns numpy arrays (~48 B/bar; 1 year ≈ 18 MB vs ~10x as dicts) and
> `ingest_market_data(..., persist=False)` transforms in memory without writing parquet.
>
> **Note (import layout):** QiMera ships its own `app` package next to Meruem's `backend/app`.
> QiMera modules import their own code as `qimera.app.*` and Meruem services via the top-level
> `app.*` (resolving to `backend/app` when `backend/` precedes `qimera/` on `sys.path` — the qimera
> test `conftest.py` and `start.ps1` (`PYTHONPATH=repo_root;qimera`) both guarantee that). Don't
> mix `backend.app.*` imports into qimera: Meruem's internal imports are bare `app.*`, and loading
> the same module under two names double-registers SQLModel tables.

### Phase M2: Python/Julia Interop Layer Proof-of-Concept
- [x] Create stdlib-only Julia package under [`qimera/julia/src/QiMera.jl`](file:///C:/Users/geral/Meruem/qimera/julia/src/QiMera.jl).
- [x] Implement EMA cross trailing-stop fitness logic in Julia.
- [x] Implement nested-loop rolling min/max sweep stats in Julia.
- [x] Build [`qimera/app/julia_bridge.py`](file:///C:/Users/geral/Meruem/qimera/app/julia_bridge.py) using `juliacall`.
- [x] Implement numerically-identical Python reference twins for validation.
- [x] Add `/api/julia/status` and `/api/benchmark` endpoints to [`qimera/app/main.py`](file:///C:/Users/geral/Meruem/qimera/app/main.py).
- [x] Verify mathematical parity (`< 1e-9`) and benchmark 4–5× speedups via tests.

### Phase M3: Single-Asset/Indicator Parametric GP ✅
- [x] Build the Python-side strongly-typed GP orchestrator using the `DEAP` library (`gp_engine.py` — typed gene domains, int/float kinds preserved by every operator).
- [x] Integrate the Julia precompiled `fitness` function via the interop bridge into the DEAP population loop (pure-Python twin as exact fallback).
- [x] Implement the hard hurdle fitness gate (drawdown/trade count boundaries) and parsimony penalty for parameter bloat.
- [x] Run end-to-end tests for a single asset with a single indicator type (`tests/test_gp_engine.py`, 6 tests).

> `deap>=1.4.1` added to `qimera/requirements.txt`. Next milestone: **M9** — champions
> registry (SQLite schema + rollback utilities).

### Phase M4: Walk-Forward Validation & Monte Carlo Gate ✅
- [x] Implement the walk-forward splitting engine with chronological folds and a 1-day purged gap (`validation.py` — expanding train, contiguous OOS blocks, purge between train/test only).
- [x] Implement the Out-Of-Sample (OOS) performance degradation checks (PF floor, PF/WR degradation caps, drawdown ceiling).
- [x] Implement the Julia-accelerated Monte Carlo stress-testing module — 10,000 bootstrap iterations with **sign-flip null** (a plain shuffle leaves the equity sum invariant and tests nothing), Gaussian slippage, latency folded into effective slippage (`QiMera.jl mc_equities` + exact numpy twin).
- [x] Log validation run details and rejected candidates with failure causes (`ValidationLog` JSONL under `qimera/data/validation/`).

> M4 caveats: with the sign-flip null each draw has zero expectation, so the Julia↔Python parity test asserts
> *statistical* agreement (mean within 4 standard errors, std within 15%) rather than exact equality. Empty PnL
> input now returns zeros in both engines (Julia previously returned uninitialized memory).

### Phase M5: Multi-Indicator GP Framework ✅
- [x] Implement typed indicator schemas and fitness hooks for:
  - [x] VWAP (anchoring, width thresholds, session resets) — `_signal_vwap`: session-anchored VWAP re-anchored per UTC day, ATR-width band, reclaim confirmation.
  - [x] FVG (gap validation, fill invalidation, expiration) — `_signal_fvg`: 3-candle gap, ATR-scaled min gap, freshness decay, fill/expiry invalidation.
  - [x] Liquidity levels (equal high/low tolerances, sweep lookbacks) — `_signal_liquidity`: tolerance-scaled sweeps of rolling extremes with confirm smoothing.
  - [x] Order Blocks (zones, freshness decay, mitigation rules) — `_signal_order_blocks`: high-volume engulfing zones, freshness cap, ATR mitigation tolerance.
  - [x] CVD (delta thresholds, volume profiles) — `_signal_cvd`: session CVD z-score vs profile-period, tanh-scaled delta threshold, confirm bars.
- [x] Shared genotype `[indicator_id, p0..p2, atr_len, atr_mult]` with per-schema typed bounds enforced by `_fix` after every DEAP operator.
- [x] Fitness hook (`strategy_metrics`) returns the M3/M4 Julia metric contract `[net, win_rate, pf, max_dd, trades]` with ATR trailing-stop exits.
- [x] DEAP orchestration (`run_multi_gp`) with indicator-switch mutation; JSON-safe output (no numpy leaks).
- [x] 16 offline tests (`tests/test_gp_indicators.py`) incl. schema typing, signal contracts, known-behaviour checks, gates, determinism, JSON serializability.

### Phase M6: Indicator Interpretation Agent ✅
- [x] Build the bounded workflow engine (`BoundedWorkflow` — explicit node/edge state machine, step budget, audit trail; LangGraph-style boundedness without the dependency tree).
- [x] Implement LLM concept → typed M5 schema mapping (`InterpAgent`, reuses Meruem's `UniversalLLM`; deterministic keyword fallback offline).
- [x] Implement search tool usage for looking up concept definitions on the web (`WebSearchTool`, keyless Wikipedia opensearch; offline-safe → []).
- [x] Log source provenance (retrieval date, URL) for auditability (`qimera/data/interp/interpretations.jsonl`).
- [x] Wrap target code generation with the isolated sandboxed Backtest Guard (`run_backtest_guard`): AST import allowlist (numpy/pandas/math/qimera.app.gp_indicators), restricted `__builtins__` exec namespace (no `__import__`/`open`/`eval`/`compile`/`getattr`), `python -I` subprocess with stripped env + hard wall-clock timeout — runaway loops are killed.
- [x] Add `POST /api/interp/interpret` endpoint (lazy imports, offline-degradable).
- [x] 18 offline tests (`tests/test_interp_agent.py`) incl. the sandbox: valid code, forbidden imports (incl. aliasing/multi-import AST catches), builtins-dict bypass, runaway-loop kill, crash capture.

> Design note: per request, the M6 workflow engine is a lightweight pure-Python bounded state machine
> rather than the LangGraph dependency — same structure (nodes/edges/audit), swappable later.

### Phase M7: Agentic Feature Synthesis ✅
- [x] Design the Evolution Agent to query the Metadata Registry and propose mutations (`synthesis.py` — typed parameter perturbations with registry/derived bounds, normalizer swaps, and composite add/subtract/product/ratio features).
- [x] Design the Coder Agent to write vectorized NumPy/pandas expressions from primitive feature definitions (deterministic template codegen; LLM refinement validated through the M6 sandboxed guard with template fallback).
- [x] Integrate the proposals into the common validation candidate pool (`FeatureCandidatePool` — score, rank, top-k, JSON-safe records; `run_synthesis` end-to-end).
- [x] Add `POST /api/synthesis/run` endpoint.
- [x] 13 offline tests (`tests/test_synthesis.py`) incl. template↔direct parity through the sandbox guard.

### Phase M8: Unified XGBoost Classifier ✅
- [x] Implement XGBoost model training using regime labels as direct input features (`classifier.py` — regime_code + regime_confidence + GARCH/HMM probabilities in the feature matrix, always-present columns for a stable feature set).
- [x] Set up exponential sample decay weights based on configured time decay (λ) (`ClassifierConfig.decay_lambda` → `weights_for`).
- [x] Integrate low-confidence output gating when regime filter confidence is below the target threshold (`predict_gated` — withheld calls below `gate_confidence`).
- [x] Write the offline validation drift monitor to run after retraining cycles (`drift_report` per-feature Kolmogorov–Smirnov + `save_drift_report` JSONL under `qimera/data/drift/`; `rolling_validate` runs the full retrain→OOS→drift cycle per window).
- [x] Add `POST /api/classifier/validate` endpoint (503 when xgboost absent).
- [x] 9 offline tests (`tests/test_classifier.py`) incl. a learned-accuracy check on trending data and KS drift detection.

### Phase M9: Champions Registry
- [ ] Set up the local SQLite database to store all historical champion candidate metadata.
- [ ] Define the registry schema (asset, regime, source, validation run ID, promotion status, manifest version).
- [ ] Add rollback utilities to restore superseded champions.

### Phase M10: Export Artifact Packaging
- [x] Define versioned plain JSON manifest file schema.
- [x] Build mock/skeleton manifest creation utility in [`qimera/app/export_store.py`](file:///C:/Users/geral/Meruem/qimera/app/export_store.py).
- [x] Wire `POST /api/inject` to push the generated manifest into Meruem's API.
- [ ] Integrate the actual GP, synthesis output models, and validation metadata into the manifest builder.

### Phase M11: Multi-Asset Rollout & Automation
- [ ] Automate retraining schedules using APScheduler.
- [ ] Set up seed population logic from prior champions.
- [ ] Standardize the CLI runner to scale processing across multiple symbols sequentially.
