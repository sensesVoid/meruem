"""Universal LLM inference through OpenAI-compatible endpoints.

Any OpenAI-compatible provider (Gemini, Groq, OpenRouter, Together, Perplexity,
a local vLLM/LM Studio host, or a paid cloud service added later in the Vault
UI) can be registered in the ``llm_providers`` setting. Providers are tried in
list order; the local Ollama model is the final automatic fallback. If nothing
succeeds we raise :class:`AllProvidersFailed` so the caller can pause the system.
"""

from app.agents.ollama_client import OllamaClient
from app.services.model_config import get_providers, get_agent_model
from app.services.vault import VaultService


class AllProvidersFailed(Exception):
    """Raised when every configured cloud provider and local Ollama failed."""


class UniversalLLM:
    """Ordered, configurable LLM chain. Cloud providers, then local Ollama.

    ``agent_id`` (optional) names the requesting agent so the chain resolves
    that agent's assigned model for cloud calls and for the Ollama fallback
    instead of always using the shared validator model.
    """

    def __init__(self, agent_id: str | None = None):
        self._ollama = None
        self.agent_id = agent_id

    def _ollama_client(self) -> OllamaClient:
        from app.services.model_config import get as model_get

        if self._ollama is None:
            model = get_agent_model(self.agent_id) if self.agent_id else (
                model_get("ollama_validator_model")
            )
            self._ollama = OllamaClient(
                base_url=model_get("ollama_url"),
                model=model,
            )
        return self._ollama

    def generate(self, prompt: str, temperature: float = 0.1) -> tuple[str, str]:
        """Try each enabled provider, then local Ollama.

        Returns (raw_text, provider_id). Raises AllProvidersFailed on total loss.
        """
        for provider in get_providers():
            if not provider.get("enabled"):
                continue
            try:
                text = self._call_provider(provider, prompt, temperature)
                if text and text.strip():
                    return text, provider.get("id", "unknown")
            except Exception:  # noqa: BLE001 - try the next provider / fallback
                continue

        try:
            text = self._ollama_client().generate(prompt, temperature=temperature)
            return text, "ollama"
        except Exception as exc:  # noqa: BLE001
            raise AllProvidersFailed(
                f"ALL_VALIDATION_PROVIDERS_FAILED -> PAUSED_LLM_LIMIT (last error: {exc})"
            ) from exc

    def _call_provider(self, provider: dict, prompt: str, temperature: float) -> str:
        from openai import OpenAI

        key = self._provider_key(provider)
        if not key:
            raise RuntimeError(f"no API key stored for provider '{provider.get('name')}'")
        model = provider["model"]
        if self.agent_id:
            model = get_agent_model(self.agent_id) or model
        client = OpenAI(base_url=provider["base_url"], api_key=key)
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are Meruem, a concise trading assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return chat.choices[0].message.content or ""

    def _provider_key(self, provider: dict) -> str | None:
        ref = provider.get("key_ref", "")
        if not ref:
            return None
        if ":" in ref:
            namespace, username = ref.split(":", 1)
        else:
            namespace, username = "llm", ref
        return VaultService.get(namespace, username)