# Meruem — Live-Testing Checklist

Run these before transitioning to the Rust desktop build (Phase 11). Each item is **action → expected result**. Mark with ✓ or ✗; for ✗ record the exact error, browser console output, and the last lines of `logs\backend.log`.

## 🔴 Safety preamble

- [ ] Run on a **demo / small-funded account**, micro lots only
- [ ] **Fortune Prime MT5 terminal running & logged in** (needed for market data + master connection)
- [ ] Backup first: `copy backend\meruem.db backend\meruem.db.bak`
- [ ] Note baseline: current `backend\meruem.db` size, open MT5 positions, account equity

---

## A. Boot & pre-flight

- [ ] `.\start.ps1` → backend healthy (`http://127.0.0.1:8765/api/health` = ok), frontend loads, TopNav **no** "BACKEND OFFLINE"
- [ ] Restart backend mid-session → TopNav shows OFFLINE then recovers green once up

## B. Brokers tab

- [ ] Connect **master** MT5 (login/server/password) → "MT5 connection OK", account saved as master, appears in Brokers + Dashboard
- [ ] Connect a **second master** → first is auto-demoted to slave (single-master enforcement)
- [ ] Connect **1–2 slave** MT5 accounts (or Binance/MEXC with API keys) → status shows connected; disable toggle works and removes from fan-out
- [ ] Credentials live in **OS keyring** (Vault tab shows mt5/binance/mexc configured) — never in files/DB

## C. Chart & market data

- [ ] XAUUSD/EURUSD/XAGUSD candles come from **mt5** provider badge; BTC/ETH from **binance**
- [ ] Timeframes M1 → 1W load; switching symbols resets state and updates provider badge
- [ ] **LIVE** badge pulses during market hours; on weekend/closed → **"MARKET CLOSED · mt5"** and chart is **frozen** (no fake movement)
- [ ] After executing a trade: **ENTRY / SL / TP overlay lines** render on the charted symbol; SL line ratchets with trailing stop; hover shows **tooltip** (entry, managed SL, lot, opened, last price)

## D. Dashboard & copier

- [ ] Master card: funded = set balance, running capital = **live equity**, gain $ and %, open/closed trades, realized PnL
- [ ] Slave rows show running capital with green ● when connected
- [ ] **Equity curve** renders with sane stats (win rate, profit factor, max drawdown)
- [ ] Fan-out simulator: proportional / fixed / risk-% lot modes produce correct per-account lots

## E. Strategies tab (full pipeline)

- [ ] **Save Reference** (name + concept only) → saved, count increments
- [ ] **Synthesize & Optimize** → stages flow synth → backtest → refine → gate; diagnostics show grounding score, IR blueprint, self-check, **Monte Carlo scores**
- [ ] Add a second concept → re-synthesize **compiles across the whole reference library**
- [ ] **Approve** → status active; scanner/validator now consume it
- [ ] Backtest with **exact date window** from extracted dataset; coverage strip shows ✓ covered / ○ missing
- [ ] Give a vague/contaminated concept → pipeline fails the gate with **recommendations** (not a silent pass)
- [ ] **LTR**: with unreviewed losing trades, run LTR → journal-of-error + review per loss; re-run → **skips already-reviewed** (no duplicates)

## F. Signals & scan cycle

- [ ] **Scan now** → per-watchlist analysis; detail shows local scanner analysis + cloud validation + trade-plan levels
- [ ] Auto-scan toggle + interval **persist across restart** (scheduler honors saved prefs)
- [ ] **Auto-execute OFF** → clicking "Accept & Execute" is the *only* path; nothing fires automatically even if verdict is ACCEPTABLE
- [ ] **Auto-execute ON** → accepted signals fan out to master + slaves automatically (subject to risk gate)
- [ ] Executed trade appears in Trades tab **with symbol/broker/order-id** and is **confirmed in MT5** — no phantom rows
- [ ] BTCUSD (or any second asset) valid setup → click Accept → **no 500 error**
- [ ] During a news block window → scan short-circuits (`news_blocked`) and trades are blocked

## G. Risk enforcement

- [ ] Set `max_concurrent_auto_trades = 1`, open 2 → second auto trade blocked with reason
- [ ] Lower `daily_loss_cap_pct` to −1% → a losing day disengages auto-mode (status shows it)
- [ ] Set `max_drawdown_pct` tight → all auto-execution blocked until cleared
- [ ] Risk tab: plan, clearance calculator, advisory render correctly

## H. ML Scanner & datasets

- [ ] Extract datasets for **multiple watchlist symbols at once**, exact start/end dates → parquet rows + date range correct
- [ ] Coverage indicator matches (covered/missing per timeframe)
- [ ] Train → status `ready`; metrics + feature importance render; model file exists under `backend/data/models/`
- [ ] Backtest on a date window reads the **extracted parquet** (not live fetch)

## I. Vault & agents

- [ ] Add cloud keys (Gemini/Groq) → `/agents/status` shows key_set + ready
- [ ] Per-agent models: lock/unlock, per-row save; catalog reflects the assignment; unsaved edits don't take effect
- [ ] Ollama reachable; model list loads

## J. Persistence & cleanup

- [ ] Restart app → accounts, strategies, settings all persist; scheduler resumes with saved interval
- [ ] `POST /api/cleanup/run` dry-run first, then real → old scans/backtests pruned per `data_retention_days`

---

## Defaults reference (from `settings_service.py`)

| Setting | Default |
|---------|---------|
| `max_concurrent_auto_trades` | 2 |
| `daily_loss_cap_pct` | −3.0 |
| `max_drawdown_pct` | −8.0 |
| `master_lot_sizing_mode` / `master_risk_pct` | `risk_pct` / 1.0 |
| `auto_execution_enabled` | false |
| `auto_scan_enabled` / `scan_interval_seconds` | true / 300 |
| `news_block_before_min` / `news_block_after_min` | 30 / 45 |
| `data_retention_days` | 90 |

## Reporting failures

For every ✗, collect:
1. What you did (tab, button, inputs)
2. Expected vs actual result
3. Browser console errors (F12 → Console)
4. Last ~25 lines of `logs\backend.log`
5. Whether a restart reproduces it
