# OpenCode Build Blueprint: Project Meruem

## 1. Environment & Technical Stack
- **OS/Host:** Windows/Linux (Acer Aspire 7, 16GB RAM, RTX 3050 Ti 4GB VRAM)
- **Primary Model (Builder):** DeepSeek-V4 Flash via OpenCode CLI
- **Frontend Core:** Tauri 2.x + React + TypeScript + Tailwind CSS + shadcn/ui
- **Frontend Charting:** `@tradingview/lightweight-charts` (MIT open source)
- **Backend Core:** FastAPI (Python sidecar via Uvicorn), SQLite (SQLModel/SQLAlchemy)
- **AI Stack:** Ollama (Scanner: local `qwen2.5:3b-instruct` or `phi3.5`), Gemini 2.0 Flash (Validator), ChromaDB (RAG Vector Store)
- **Brokers & Copier:** MetaTrader5 Python API, `ccxt` (Binance), OS Native `keyring` (Vault)

---

## 2. Core Operational Configs & Risk Caps (Fully User-Configurable)

> [!IMPORTANT]  
> All risk thresholds, drawdown limits, trade caps, and lot sizing parameters are **100% user-configurable** via the UI Settings & Vault panel. They are stored in SQLite (`settings` table) and dynamically evaluated by the backend Risk Interceptor prior to order placement.

### A. Dynamic Risk Guardrails & Safety Ceilings
* **Max Concurrent Auto-Trades:** Configurable integer (Default: `2` global trades across master and slave accounts).
* **Daily Hard Loss Cap (Auto-Disable Switch):** Configurable % or dollar drop from starting daily equity (Default: `-3.0%`). Reaching this ceiling disengages auto-trading mode instantly.
* **Max Total Drawdown Floor:** Configurable % or dollar overall equity drop from initial balance (Default: `-8.0%`). Reaching this floor disengages auto-mode and blocks all automated trade execution.

### B. Master & Slave Lot Sizing Engine (Configurable Modes)
* **Master Lot Sizing:** Master account calculates lot size based on user-configured account risk % or manual lot input.
* **Slave Account Scaling Modes (Selectable per Slave Account):**
  1. **Proportional Equity Sizing (Default):**  
     $$\text{Slave Lot} = \text{Master Lot} \times \left(\frac{\text{Slave Equity}}{\text{Master Equity}}\right) \times \text{User Scale Factor}$$
  2. **Fixed Multiplier Sizing:**  
     $$\text{Slave Lot} = \text{Master Lot} \times \text{Lot Multiplier}$$
  3. **Custom Risk % Override:** Calculates lot size independently on each slave account based on slave equity and trade Stop Loss distance:
     $$\text{Slave Lot} = \frac{\text{Slave Equity} \times \text{Custom Risk \%}}{\text{SL Distance (pips)} \times \text{Pip Value}}$$

### C. Audit Agent "Stable Threshold" Profiles (User-Selectable & Customizable)
* **Prop Firm Challenge Baseline (Default Preset):**
  * Win Rate Floor: `45%`
  * Profit Factor Floor: `1.5`
  * Max Drawdown Ceiling: `5.0%`
* **Conservative (Funded / Live Account Preset):**
  * Win Rate Floor: `50%`
  * Profit Factor Floor: `1.8`
  * Max Drawdown Ceiling: `4.0%`
* **Aggressive (Phase 1 Target Push Preset):**
  * Win Rate Floor: `40%`
  * Profit Factor Floor: `1.3`
  * Max Drawdown Ceiling: `7.0%`
* **Custom Preset:** Fully customizable Win Rate, Profit Factor, and Drawdown parameters set via UI.

### D. Validation Provider Fallback Logic
* **Primary:** Gemini 2.0 Flash (`google-generativeai`)
* **Secondary Fallback:** Groq API / DeepSeek API free tier
* **Tertiary Fallback:** Local Ollama model (`qwen2.5-coder:7b`)
* **Fail-Safe Behavior:** If all validation providers fail or hit rate limits, set System Status to `PAUSED_LLM_LIMIT`, display notification badge on UI, log scan output to SQLite, and freeze auto-execution until manual acknowledgement.

---

## 3. Step-by-Step Implementation Roadmap

### Phase 0: Project Scaffold
- [ ] Initialize Tauri 2.x app with React + TypeScript + Tailwind + shadcn/ui.
- [ ] Create Python FastAPI backend structure under `/backend`.
- [ ] Setup Uvicorn sidecar lifecycle inside Tauri process manager.
- [ ] Build SQLite database model using SQLModel (`accounts`, `assets`, `strategies`, `scans`, `validations`, `trades`, `audit_reports`, `settings`).
- [ ] Implement `keyring` API for encrypted storage of broker credentials.

### Phase 1: Core Scanning & Validation Engine
- [ ] Implement MT5 connection manager (connect, healthcheck, pull M1-H4 OHLCV).
- [ ] Create Ollama client to run local Scanner Agent (`qwen2.5:3b-instruct` / `phi3.5`).
- [ ] Implement Gemini 2.0 Flash Validator Agent with standard prompt template.
- [ ] Add Fallback chain (Gemini → Groq/DeepSeek → Ollama → `PAUSED_LLM_LIMIT`).
- [ ] Build UI Asset Watchlist pane and Trade Recommendation Card component.

### Phase 2: Execution, Trade Management & Multi-Account Copier
- [ ] Implement MT5 Order Placement logic (Market, Stop, Limit).
- [ ] Build Multi-Account Trade Copier Engine supporting Proportional Equity, Fixed Multiplier, and Custom Risk % modes.
- [ ] Create Position Manager for dynamic TP1–TP4 management, Trailing Stops, and Auto-Breakeven (Auto-BE when TP1 hit or price reaches user R-multiple).
- [ ] Build Settings UI & Backend Risk Interceptor (Dynamically fetches user-configured max daily loss %, total drawdown %, and max concurrent auto-trades before evaluating order placement).

### Phase 3: RAG & Strategy Library
- [ ] Initialize ChromaDB persistent client in Python backend.
- [ ] Create Strategy Library UI (CRUD operations for Markdown trading rules).
- [ ] Connect embeddings pipeline (`sentence-transformers/all-MiniLM-L6-v2` or `nomic-embed-text`).
- [ ] Inject retrieved strategy vector context into Scanner and Validator prompts.

### Phase 4: Backtester & Audit Agent
- [ ] Integrate `backtesting.py` pipeline fed with historical OHLCV data.
- [ ] Implement 7-day automated audit cycle using `APScheduler`.
- [ ] Build Audit Dashboard in UI comparing live trades against selectable/custom Stable Threshold profiles.
- [ ] Create Strategy Optimizer Agent to auto-propose strategy tweaks and queue recommendations for human approval.

---

## 4. Final Completion Verification Checklist

*Before marking the project as COMPLETE, the OpenCode agent must test and confirm every item in this checklist:*

### UI & Desktop Shell
- [ ] Tauri app launches successfully without console or Rust bridge errors.
- [ ] Layout renders correctly with resizable or clear grid panels (Charts, News, Watchlist, Trades, Strategy Library, Audit, Settings).
- [ ] Charting panel loads `@tradingview/lightweight-charts` and updates live on asset selection.

### Backend & API
- [ ] FastAPI sidecar starts automatically with Tauri and responds to healthchecks on `localhost`.
- [ ] SQLite database automatically creates tables (`accounts`, `assets`, `strategies`, `scans`, `validations`, `trades`, `audit_reports`, `settings`) and performs migrations without error.
- [ ] OS `keyring` correctly saves and retrieves test API keys/credentials without plain-text leakage.

### AI Engine & Fallback
- [ ] Scanner Agent generates valid structured JSON candidates from Ollama.
- [ ] Validator Agent returns `Acceptable` / `Not Acceptable` responses with entry, SL, and TP levels.
- [ ] Fallback system gracefully switches to backup provider or pauses system (`PAUSED_LLM_LIMIT`) if Gemini API returns 429 / offline errors.

### Trade Management & Dynamic Risk Safeguards
- [ ] UI Settings allow full customization of risk caps (Daily Loss %, Total Drawdown %, Concurrent Auto-Trades) and Lot Sizing parameters.
- [ ] Backend Risk Interceptor dynamically loads settings from SQLite and blocks auto-execution when configured caps are exceeded.
- [ ] Master account trade execution immediately fans out to slave copy-accounts with correct lot sizes matching the selected scaling mode (Proportional Equity, Fixed Multiplier, or Custom Risk %).
- [ ] Auto-Breakeven triggers instantly when trade reaches configured TP1/R-multiple threshold.

### Audit & RAG Pipeline
- [ ] Saving a strategy in the Strategy Library updates ChromaDB vector index immediately.
- [ ] `APScheduler` successfully executes audit logic and calculates Win Rate, Profit Factor, and Max Drawdown metrics against selected threshold profile.
- [ ] Strategy Optimizer proposals remain strictly in `DRAFT` / `PENDING_APPROVAL` status until explicitly approved via UI.