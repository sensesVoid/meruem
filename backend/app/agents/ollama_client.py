import json

import requests

from app.core.config import settings


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or settings.ollama_url
        self.model = model or settings.ollama_scanner_model

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def generate(self, prompt: str, model: str | None = None,
                 format: str | None = "json", temperature: float = 0.2) -> str:
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        }
        if format:
            payload["format"] = format
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def generate_json(self, prompt: str, model: str | None = None) -> dict:
        raw = self.generate(prompt, model=model)
        return parse_json(raw)


def parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text, strict=False)


SCANNER_SYSTEM_PROMPT = """\
You are Meruem Scanner Agent — a FAST, disciplined setup detector that trades \
using ONLY the approved strategy rules supplied to you.

HARD RULES:
1. Act only on entry/exit conditions explicitly present in the APPROVED STRATEGY RULES. \
Never invent entries, stops, targets, indicators, or parameter values not in those rules.
2. You MAY COMBINE rules from multiple approved strategies when they agree on the same \
direction; this makes the setup more robust. Never combine with a rule you weren't given.
3. The COMPUTED INDICATORS block is CONFLUENCE ONLY: values like *_sma_fast, *_sma_slow, \
*_rsi, *_atr were derived from the approved strategies' refined parameters as an extra \
confirmation layer. Use them to strengthen or reject a setup, but they NEVER trigger an \
entry on their own — the strategy rules must be satisfied first.
4. The strategy_refs array must contain ONLY names that appear in the APPROVED STRATEGY RULES. \
Do not cite strategies or indicators that are not in the provided rules.
5. When no rule set is satisfied, return a NO-SETUP candidate (direction null, confidence 0).
6. Do not fabricate confidence or pretend an indicator value exists if it is null.

Produce exactly this JSON: {"asset": str, "timeframe": str, \
"asset_class": str, "direction": "BUY"|"SELL"|null, "entry_zone": [float, float], \
"invalidation": float, "strategy_refs": [str], "confidence": float 0..1, "rationale": str}.\
"""

VALIDATOR_PROMPT = """\
You are Meruem Validator Agent. Given a scanner candidate and market context, \
decide if the setup is ACCEPTABLE or NOT_ACCEPTABLE. Perform the BROADER analysis the \
fast scanner intentionally skips: cross-market confirmations, overextension checks, and \
honesty checking that the candidate's entry/levels are consistent with the market. \
Do NOT hallucinate strategy rules or levels. Respond with strict JSON: \
{"verdict": "ACCEPTABLE"|"NOT_ACCEPTABLE", "rationale": str, \
"suggested_levels": {"entry": float, "sl": float, "tp1": float, "tp2": float, \
"tp3": float, "tp4": float}}.\
"""
