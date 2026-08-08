import json
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from app.models.entities import Setting

DEFAULT_SETTINGS: dict[str, Any] = {
    "max_concurrent_auto_trades": 2,
    "daily_loss_cap_pct": -3.0,
    "max_drawdown_pct": -8.0,
    "master_lot_sizing_mode": "risk_pct",  # risk_pct | manual
    "master_risk_pct": 1.0,
    "master_manual_lot": 0.10,
    "selected_master_account_id": "",
    "default_account_equity": 10000.0,
    "auto_execution_enabled": False,
    "auto_scan_enabled": True,
    "scan_interval_seconds": 300,
    "data_retention_days": 90,  # prune scans/backtests/closed-trades older than this (0 = keep all)
    "system_status": "READY",  # READY | PAUSED_LLM_LIMIT | PAUSED_RISK | DISABLED
    # LLM / local model selection (used by scanner + validator chain).
    #
    # `llm_providers` is a dynamic, universal list of OpenAI-compatible LLM
    # endpoints. Each entry: {id, name, base_url, model, enabled, key_ref}.
    # Providers are tried in list order during validation / refinement / QA;
    # the local Ollama model is the final automatic fallback. The list is
    # user-editable in the Vault UI, so any paid/cloud LLM service with an
    # OpenAI-compatible API can be added later without code changes.
    "llm_providers": [
        {
            "id": "gemini",
            "name": "Gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": "gemini-3.5-flash-lite",
            "enabled": True,
            "key_ref": "llm:gemini",
        },
        {
            "id": "groq",
            "name": "Groq",
            "base_url": "https://api.groq.com/openai/v1",
            "model": "llama-3.3-70b-versatile",
            "enabled": True,
            "key_ref": "llm:groq",
        },
    ],
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_scanner_model": "qwen2.5:3b-instruct",
    "ollama_validator_model": "qwen2.5:7b-instruct",
    "tp_config": {
        "tp1": 1.0,
        "tp2": 2.0,
        "tp3": 3.0,
        "tp4": 4.0,
    },
    # Task #12: per-agent LLM model assignment, keyed by agent id (scanner,
    # validator, regime, news, ingestion, synthesizer, backtest, refiner,
    # ltr, qa, refinement, risk_manager, xgb_scanner). Blank = catalog default.
    "agent_models": {},
    "auto_breakeven_enabled": True,
    "auto_be_trigger": "tp1",  # tp1 | r_multiple
    "auto_be_r_multiple": 1.0,
    "trailing_stop_enabled": True,
    "trailing_stop_activation_r": 1.5,
    "trailing_stop_step_pips": 10.0,
    # News trade-block window (phase 2): minutes BEFORE and AFTER a high-impact
    # event's publish time during which auto scan/trades are blocked. UI range 15-60.
    "news_block_before_min": 30,
    "news_block_after_min": 45,
}


class SettingsService:
    def __init__(self, session: Session):
        self.session = session

    def _ensure_defaults(self) -> None:
        for key, value in DEFAULT_SETTINGS.items():
            existing = self.session.exec(
                select(Setting).where(Setting.key == key)
            ).first()
            if existing is None:
                self.session.add(Setting(id=key, key=key, value_json=json.dumps(value)))
        # Prune legacy per-provider model rows replaced by the `llm_providers` list.
        for legacy in ("llm_model_gemini", "llm_model_groq", "llm_model_deepseek", "llm_model_openai"):
            row = self.session.exec(select(Setting).where(Setting.key == legacy)).first()
            if row is not None:
                self.session.delete(row)
        self.session.commit()

    def get_all(self) -> dict[str, Any]:
        self._ensure_defaults()
        rows = self.session.exec(select(Setting)).all()
        return {r.key: json.loads(r.value_json) for r in rows}

    def get(self, key: str, default: Any = None) -> Any:
        row = self.session.exec(select(Setting).where(Setting.key == key)).first()
        if row is None:
            return default
        return json.loads(row.value_json)

    def set(self, key: str, value: Any) -> None:
        row = self.session.exec(select(Setting).where(Setting.key == key)).first()
        if row is None:
            self.session.add(
                Setting(id=key, key=key, value_json=json.dumps(value))
            )
        else:
            row.value_json = json.dumps(value)
            row.updated_at = datetime.now(timezone.utc)
        self.session.commit()

    def reset_to_defaults(self) -> None:
        for key, value in DEFAULT_SETTINGS.items():
            self.set(key, value)