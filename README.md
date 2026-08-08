# Meruem — AI-Agentic Trading Desktop

Local-first automated trading terminal. Tauri desktop shell + FastAPI Python sidecar, driven by a multi-agent AI pipeline (local Ollama scanner → cloud-validator fallback chain) with dynamic risk interception, master–slave lot copier, RAG strategy library, and a 7-day audit agent.

> Built per `Instructions.md` and `PRD.md` (v0.2).

## Stack

| Layer | Tech |
|-------|------|
| Desktop shell | Tauri 2.x (Rust) |
| Frontend | React 18 + TypeScript + Tailwind CSS + Vite |
| Charting | `lightweight-charts` (TradingView MIT) |
| Backend | FastAPI + Uvicorn sidecar on `127.0.0.1:8765` |
| Database | SQLite via SQLModel (8 tables) |
| AI | Ollama scanner + Gemini 2.0 Flash → Groq → DeepSeek → Ollama validator chain |
| RAG | ChromaDB + `sentence-transformers` |
| Credentials | OS `keyring` |
| Scheduler | APScheduler (7-day audit, periodic scan) |
| Brokers | `MetaTrader5`, `ccxt` (Binance) |

## Repo Layout

```
backend/
  app/
    core/        config, db (SQLModel engine + init)
    models/      entities.py (8 SQL tables)
    agents/      ollama client, ScannerAgent, ValidatorChain (Gemini/Groq/DeepSeek/Ollama)
    services/    settings, vault, risk_interceptor, copier, position_manager,
                 rag, audit, scheduler, orchestrator, mt5_adapter, binance_adapter
    api/         FastAPI REST router
  tests/         pytest integration suite
frontend/
  src/           React app: panels, chart, api client
  vite.config.ts proxies /api -> 127.0.0.1:8765
src-tauri/       Rust shell (requires Rust toolchain to build)
start.ps1        Launches backend + frontend together
```

## Prerequisites

- Python 3.12+
- Node 20+ / npm
- Rust toolchain (only needed to build the Tauri desktop shell — not required to run `start.ps1` in browser)

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

Open http://localhost:1420 in a browser.

UI panels: **Chart** (lightweight-charts, live-simulated ticks), **Asset Watchlist**,
**News & Economic Calendar** (persistent right pane — CNBC + Investing.com RSS, plus
GDELT/CryptoPanic when available; no assets required), **Trades / Position Manager**,
**Strategy Library**, **Audit Dashboard**, **Settings & Vault**. If the top bar shows
`BACKEND OFFLINE`, the backend isn't running — start it with `.\start.ps1`.

Desktop shell (once Rust is installed):

```powershell
cd frontend; npm run tauri dev
```

## API (prefix `/api`)

- `GET /health` — backend + DB health
- `GET /status` — system status, auto-execution, live risk gate
- `GET|PUT /settings`, `POST /settings/reset` — user-configurable risk caps & lot sizing (applied instantly, no restart)
- `POST|GET /vault/...` — store/retrieve secrets in OS keyring
- `/accounts`, `/assets`, `/strategies` (CRUD + RAG embed/search), `/scans`, `/validations`, `/trades`, `/audit/*`, `/risk/check`

## Risk Guardrails (user-configurable, enforced live)

- Max concurrent auto-trades (default `2`)
- Daily hard loss cap (default `-3.0%`), breach disengages auto-mode
- Max total drawdown ceiling (default `-8.0%`), breach blocks all auto-execution
- Every order passes through `RiskInterceptor` which reloads settings from SQLite per evaluation.

## Fallback Fail-Safe

Gemini → Groq → DeepSeek → local Ollama → **`PAUSED_LLM_LIMIT`** (log to DB, set System Status, halt auto-execution until manual acknowledge).