# Meruem — AI-Agentic Trading Desktop

Local-first automated trading terminal: a Tauri desktop shell wrapping a React + Vite frontend and a FastAPI sidecar, driven by a multi-agent AI pipeline.

The workflow is concept-in, trades-out: ingest trading concepts → synthesize executable entry strategies strictly grounded in them → pass a merged QA gate (deterministic + LLM self-check) → scan a watchlist for live setups → AI-validate every signal → block unsafe trades with live risk interception → copy accepted trades from a master account to slave accounts across MT5 / Binance / MEXC.

> Feature roadmap and phase status live in `Pending.md`.

## Highlights

- **Concept → Strategy pipeline** — save a trading concept (e.g. *"liquidity sweep into a Fair Value Gap with a Change in State of Delivery"*) and the Synthesizer agent compiles it into a concrete entry model grounded in the ingested material. The merged QA gate (deterministic + LLM self-check) decides PASS / FAIL, and the pipeline also emits **proxy parameters** (fast/slow/RSI/ATR periods) that the scanner reads to compute real indicators at scan time.
- **Raw-LLM scanner** — the watchlist scanner reasons over every approved strategy's rules + proxy params plus the reference library and any attached ML artifact context files, producing setups from real computed indicators.
- **IR primitive registry** — entry rules are expressed with math-checkable primitives (FVG, VWAP, volume profile, etc.) rather than free text. New tools can be proposed and registered at runtime; the registry hot-swaps for the running process.
- **Live risk interception** — every auto trade passes a risk gate (concurrent-trade cap, daily hard-loss cap, total drawdown ceiling, news-block windows) before it ever reaches a broker.
- **Master–slave copier** — fan-out with proportional / fixed / risk-% lot scaling, and a position manager with auto-breakeven and ratcheting trailing stops rendered live on the chart (entry/SL/TP overlay lines with hover tooltips).
- **Universal broker connector** — one protocol (`BrokerAdapter`) implemented for MT5 (native), Binance and MEXC (via `ccxt`); adding a new broker is a single registry entry with zero call-site changes.
- **ML context files** — reference documents (`.txt`/`.md`/`.json`/`.csv`) can be uploaded per symbol and are passed to the scanner as read-only context, without any model training.
- **Local-first** — everything runs on your machine; API secrets live in the OS keyring, and the SQLite database never leaves your disk.

## Stack

| Layer | Tech |
|-------|------|
| Desktop shell | Tauri 2.x (Rust) |
| Frontend | React 18 + TypeScript + Tailwind CSS + Vite |
| Charting | `lightweight-charts` (TradingView MIT) |
| Backend | FastAPI + Uvicorn sidecar on `127.0.0.1:8765` |
| Database | SQLite via SQLModel (8 tables) |
| AI | Local Ollama scanner → configurable cloud validator chain (Gemini → Groq → DeepSeek → Ollama) |
| RAG | ChromaDB + `sentence-transformers` (keyword-search fallback without ChromaDB) |
| Brokers | `MetaTrader5` (native), `ccxt` (Binance, MEXC) |
| Credentials | OS `keyring` |
| Scheduler | APScheduler (periodic scans, LTR review, data cleanup) |

## Repo layout

```
backend/
  app/
    core/        config (env vars), db (SQLModel engine + migrations)
    models/      entities.py (8 SQL tables)
    agents/      scanner, validator chain, synthesizer, LTR agent,
                 universal LLM client
    services/    broker_base/adapters/mexc, copier, position_manager,
                 risk_interceptor, risk_manager, strategy_pipeline, ir (+primitives),
                 ml_artifact, market_data, rag, news (+agent), regime, scheduler,
                 orchestrator, vault, ...
    api/         FastAPI REST router (prefix /api)
  scripts/       poc_faithful_orderflow.py, bench_refactor.py (perf harness)
  tests/         pytest integration suite
frontend/
  src/           React app: 9 tab panels, chart, api client
  vite.config.ts proxies /api -> 127.0.0.1:8765
src-tauri/       Rust shell (requires Rust toolchain to build)
start.ps1        Launches backend + frontend together
setup.ps1        One-shot environment setup
```

## Prerequisites

- Python 3.12+
- Node 20+ / npm
- Rust toolchain (only needed to build the Tauri desktop shell — not required to run `start.ps1` in a browser)
- [Ollama](https://ollama.com) with a local model for the scanner (default `qwen2.5:3b-instruct`)

## Setup

Quick start (creates venv, installs deps, optionally ChromaDB):

```powershell
.\setup.ps1            # core deps + opt-in ChromaDB RAG
.\setup.ps1 -InitDb    # also create the SQLite schema
```

Manual equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r .\backend\requirements.txt
pip install -r .\backend\requirements-chromadb.txt   # OPTIONAL: ChromaDB RAG (see note below)
pytest .\backend\tests                               # run integration tests

cd frontend
npm install
npm approve-scripts esbuild     # if allowed-scripts blocks the esbuild postinstall
```

> **ChromaDB on Windows:** every stable `chromadb` pins `chroma-hnswlib==0.7.6`, which has no
> cp312 Windows wheel and needs MSVC++ Build Tools to compile. `setup.ps1` installs the prebuilt
> `chroma-hnswlib==0.7.5` wheel + `chromadb --no-deps` instead. Without ChromaDB, the RAG layer
> automatically falls back to keyword search — the backend runs fine either way.
> The heavy AI deps (`sentence-transformers`, `google-generativeai`, `onnxruntime`) are imported
> lazily, so the app also runs before they finish installing.

## Run

```powershell
.\start.ps1           # backend (8765) + frontend (1420) together
.\start.ps1 -BackendOnly
.\start.ps1 -FrontendOnly
```

Open http://localhost:1420 in a browser. If the top bar shows `BACKEND OFFLINE`, the backend isn't running.

### UI panels

| Tab | What it does |
|-----|--------------|
| **Dashboard** | Master account performance, equity curve (Tradezella-style with drawdown/win-rate stats), slave account running capital, and the copier fan-out simulator |
| **Chart** | Candlestick chart with watchlist (left), News & economic calendar + Market Regime (right). LIVE/MARKET CLOSED badge — synthetic ticks only animate while the market is open. Open trades render ENTRY / managed-SL / TP overlay lines with hover tooltips |
| **Signals** | Latest AI scan per watchlist asset: scanner analysis (local), validation verdict (cloud), trade plan levels, accept-and-execute |
| **Trades** | Position manager: open/closed trades, managed SL projection, close/delete, phantom-trade cleanup |
| **Brokers** | Connect MT5 / Binance / MEXC accounts, designate master/slave, live connection status, keyring-backed credentials |
| **Risk Settings** | Risk caps (concurrent trades, daily loss, drawdown ceiling), news-block window, auto-execution toggles |
| **Strategies** | Strategy library + synthesis: ingest concepts, run the pipeline (Synthesis → QA → finalize), view the QA gate verdict, proxy parameters, diagnostics and the synthesized entry |
| **Vault** | API keys for LLM providers + per-agent model assignment (locked fields with explicit save) |

Desktop shell (once Rust is installed):

```powershell
cd frontend; npm run tauri dev
```

## Configuration

Backend config is read from environment variables (`backend/.env`, prefix `MERUEM_`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `MERUEM_DATABASE_URL` | `sqlite:///backend/meruem.db` | SQLite location |
| `MERUEM_HOST` / `MERUEM_PORT` | `127.0.0.1` / `8765` | API bind address |
| `MERUEM_GEMINI_API_KEY` / `MERUEM_GROQ_API_KEY` | — | Cloud validator providers |
| `MERUEM_OLLAMA_URL` | `http://127.0.0.1:11434` | Local LLM endpoint |
| `MERUEM_OLLAMA_SCANNER_MODEL` | `qwen2.5:3b-instruct` | Local scanner model |
| `MERUEM_SCAN_INTERVAL_SECONDS` | `300` | Auto-scan cadence |
| `MERUEM_DATA_RETENTION_DAYS` | `90` | Cleanup retention |
| `MERUEM_LTR_REVIEW_THRESHOLD` | `10` | Losing trades before auto-review |

Secrets (broker passwords, API keys) are stored in the **OS keyring** via the Vault tab — never in files. LLM provider selection and per-agent model assignment are managed in the Vault tab and read live by the agents.

## API (prefix `/api`)

- **Health/status** — `GET /health`, `GET /status`
- **Settings & vault** — `GET|PUT /settings`, `POST /settings/reset`, `POST|GET|DELETE /vault/...`, `GET /vault/status`
- **Agents & LLM** — `GET /agents/status`, `GET /agents/catalog`, `GET /providers/{id}/models`, `GET|POST /ollama/models|test`
- **Brokers & accounts** — `POST /brokers/connect|test`, `GET /brokers/status`, `GET|POST|PUT|DELETE /accounts`, `GET /accounts/equity`, `POST /copier/plan`
- **Performance** — `GET /dashboard` (equity curve + per-account stats)
- **Watchlist** — `GET|POST /assets`, `DELETE /assets/{id}`
- **Strategies** — CRUD, `POST /strategies/reference`, `POST /strategies/qa`, `POST /strategies/submit` (+ job status), `POST /strategies/{id}/approve|review`, `POST /strategies/search`, `POST /strategies/ltr`, `GET /strategies/ltr/records`, `GET|POST /strategies/primitives`, `GET|POST /strategies/ml-artifacts`, `GET|DELETE /strategies/ml-artifacts/{id}`
- **Signals** — `GET /signals`, `POST /signals/scan`, `GET|POST /signals/automation`, `POST /signals/{scan_id}/accept`
- **Scans / validations / trades** — `GET /scans`, `GET /validations`, `GET|POST /trades`, `PUT /trades/{id}/close`, `DELETE /trades/{id}`, `POST /trades/clear-phantoms`, `GET /chart/positions`
- **Risk** — `GET /risk/check`, `GET /risk/plan`, `POST /risk/clear`
- **Market data & news** — `GET /candles`, `GET /news`, `GET /news-agent/state`, `POST /news-agent/refresh`, `GET /regime`, `POST /regime/refresh`
- **Maintenance** — `POST /cleanup/run`

## Risk guardrails (user-configurable, enforced live)

- Max concurrent auto-trades (default `2`)
- Daily hard loss cap (default `-3.0%`) — breach disengages auto-mode
- Max total drawdown ceiling (default `-8.0%`) — breach blocks all auto-execution
- News-block windows — trades are blocked before/after high-impact events (default 30/45 min)
- Every order passes through `RiskInterceptor`, which reloads settings from SQLite per evaluation

## Fallback fail-safe

Gemini → Groq → DeepSeek → local Ollama → **`PAUSED_LLM_LIMIT`** (log to DB, set System Status, halt auto-execution until manually acknowledged).

## Tests & benchmarks

```powershell
.\.venv\Scripts\activate
cd backend
python -m pytest tests/ -q          # 201 tests: agents, API, brokers, copier, risk, strategy pipeline, IR, ...
python scripts/bench_refactor.py    # dashboard + signals N+1→grouped performance benchmark
```

## Disclaimer

This software is for research and education. Trading involves substantial risk; no part of this project is financial advice. Use at your own risk.
