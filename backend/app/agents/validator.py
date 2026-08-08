from app.agents.ollama_client import VALIDATOR_PROMPT, parse_json
from app.agents.universal_llm import AllProvidersFailed, UniversalLLM


class ValidationFailed(Exception):
    """Raised when every LLM provider (cloud + local Ollama) is unavailable."""


class ValidatorChain:
    """Dynamic validation chain driven by the universal provider list.

    Providers are configured in the ``llm_providers`` setting (Vault UI) and
    tried in order; local Ollama is the automatic final fallback. On total
    failure a :class:`ValidationFailed` is raised so callers can enter the
    ``PAUSED_LLM_LIMIT`` fail-safe.
    """

    def __init__(self, agent_id: str | None = None):
        self.llm = UniversalLLM(agent_id=agent_id or "validator")

    def generate(self, prompt: str) -> tuple[str, str]:
        try:
            return self.llm.generate(prompt)
        except AllProvidersFailed as exc:
            raise ValidationFailed(str(exc)) from exc

    def validate(self, candidate: dict, context: dict) -> tuple[dict, str]:
        rules = context.get("strategies") or []
        rules_block = (
            "Approved strategy rules that were combined into this candidate:\n"
            f"{rules}\n\n"
            if rules else ""
        )
        references = context.get("references") or []
        refs_block = ""
        if references:
            refs_block = (
                "Reference concepts library (context only — knowledge source, never "
                "an execution rule):\n"
                f"{references}\n\n"
            )
        prompt = (
            f"{VALIDATOR_PROMPT}\n\n"
            f"Candidate: {candidate}\n"
            f"Market context: {context.get('market')}\n"
            f"{rules_block}"
            f"{refs_block}"
            "Verify the candidate does not contradict these approved rules and the "
            "levels make sense against the market. Answer only JSON."
        )
        try:
            raw, provider = self.llm.generate(prompt)
            return parse_json(raw), provider
        except AllProvidersFailed as exc:
            raise ValidationFailed(str(exc)) from exc
