import json

from app.agents.ollama_client import SCANNER_SYSTEM_PROMPT, OllamaClient
from app.services.model_config import get as model_get
from app.services.model_config import get_agent_model


class ScannerAgent:
    """Produces structured JSON candidate setups using the local Ollama model.

    The scanner is deliberately a FAST, rule-grounded detector: it maps concrete
    CANDIDATE signals (derived from the approved strategies' rules + params) into a
    single setup. All deeper cross-agreement reasoning is left to the Validator.
    """

    def __init__(self):
        self.client = OllamaClient(
            base_url=model_get("ollama_url"),
            model=get_agent_model("scanner") or model_get("ollama_scanner_model"),
        )

    def scan(self, ohlcv: dict, indicators: dict, strategies: list[dict],
             references: list[dict] | None = None,
             ml_context: list[dict] | None = None) -> dict:
        rules = json.dumps(strategies, default=str)[:8000]
        refs_block = ""
        if references:
            refs_block = (
                "\nREFERENCE CONCEPTS LIBRARY (context only — NOT executable rules):\n"
                f"{json.dumps(references, default=str)[:4000]}\n"
            )
        ml_block = ""
        if ml_context:
            ml_block = (
                "\nML ARTIFACT CONTEXT FILES (context only — human notes / model "
                "insights uploaded by the trader; use them to sharpen the read, "
                "never to contradict the approved rules):\n"
                f"{json.dumps(ml_context, default=str)[:6000]}\n"
            )
        user = (
            f"Asset: {ohlcv.get('symbol')} ({ohlcv.get('asset_class', '')})\n"
            f"Timeframe: {ohlcv.get('timeframe')}\n"
            f"Latest OHLCV bars: {json.dumps(ohlcv.get('rows', [])[-8:])}\n"
            f"Computed indicators: {json.dumps(indicators)}\n"
            f"APPROVED STRATEGY RULES (the ONLY rules you may act on):\n{rules}\n"
            f"{refs_block}{ml_block}"
            "Produce the candidate JSON now."
        )
        prompt = f"{SCANNER_SYSTEM_PROMPT}\n\n{user}"
        try:
            return self.client.generate_json(prompt)
        except Exception as exc:  # noqa: BLE001
            return {
                "asset": ohlcv.get("symbol"),
                "direction": None,
                "confidence": 0.0,
                "rationale": f"Scanner error: {exc}",
                "error": True,
            }

    def normalize(self, candidate: dict) -> dict:
        """Clamp a scanner candidate so it can never reference rules/params that
        were not actually provided. This is the hallucination firewall: any
        strategy_ref quoting a rule set that was never approved gets stripped."""
        refs = candidate.get("strategy_refs") or []
        if isinstance(refs, str):
            refs = [refs]
        candidate["strategy_refs"] = [r for r in refs if isinstance(r, str)]
        return candidate

    @staticmethod
    def key_discriminator() -> str:
        return "ollama"  # local scanner has no external key

    def has_model(self) -> bool:
        try:
            return len(self.client.list_models()) > 0
        except Exception:  # noqa: BLE001 - local Ollama absent is "no model"
            return False