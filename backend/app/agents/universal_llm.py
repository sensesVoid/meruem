"""Universal LLM inference through OpenAI-compatible endpoints.

Any OpenAI-compatible provider (Gemini, Groq, OpenRouter, Together, Perplexity,
a local vLLM/LM Studio host, or a paid cloud service added later in the Vault
UI) can be registered in the ``llm_providers`` setting. Providers are tried in
list order; the local Ollama model is the final automatic fallback. If nothing
succeeds we raise :class:`AllProvidersFailed` so the caller can pause the system.

Model routing (per-agent models are the absolute basis):

1. If the agent's assigned model is an Ollama model (installed locally), that
   model is used directly on the local Ollama instance.
2. Otherwise the provider that *configures* that model is preferred; the agent
   model is also attempted on the remaining providers.
3. If no model is assigned, each provider is tried in list order with its OWN
   configured model (the legacy provider-chain behaviour).
4. Every path falls back to a local -INSTALLED- Ollama model (never a cloud-only
   model name, which would 404 on Ollama).
"""

import time

from app.agents.ollama_client import OllamaClient
from app.services.model_config import get_providers, get_agent_model
from app.services.vault import VaultService

_ollama_cache: dict[str, tuple[float, list[str]] | None] = {}
_OLLAMA_CACHE_TTL = 10.0  # seconds


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
        self._last_error: Exception | None = None

    def _ollama_client(self) -> OllamaClient:
        from app.services.model_config import get as model_get

        if self._ollama is None:
            self._ollama = OllamaClient(
                base_url=model_get("ollama_url"),
            )
        return self._ollama

    def _ollama_installed(self) -> list[str]:
        """Cached list of models currently installed on Ollama (TTL-cached)."""
        now = time.time()
        hit = _ollama_cache.get("installed")
        if hit and now - hit[0] < _OLLAMA_CACHE_TTL:
            return hit[1]
        try:
            self._ollama_client()
            models = self._ollama_client().list_models()
        except Exception as exc:  # noqa: BLE001 - Ollama may be down
            self._last_error = exc
            _ollama_cache["installed"] = (now, [])
            return []
        _ollama_cache["installed"] = (now, models)
        return models

    def _ollama_fallback_model(self) -> str:
        """A local model that is actually installed (never a cloud model name).

        Preference: the configured validator model -> the scanner model -> the
        first installed model -> the legacy default.
        """
        from app.services.model_config import get as model_get

        installed = self._ollama_installed()
        candidates = [
            model_get("ollama_validator_model"),
            model_get("ollama_scanner_model"),
        ]
        for c in candidates:
            if c and c in installed:
                return c
        if installed:
            return installed[0]
        return model_get("ollama_validator_model") or "qwen3:4b"

    def generate(self, prompt: str, temperature: float = 0.1) -> tuple[str, str]:
        """Honour the per-agent model, chain the providers, fall back to Ollama.

        Returns (raw_text, provider_id). Raises AllProvidersFailed on total loss.
        """
        providers = [p for p in get_providers() if p.get("enabled")]
        assigned = (get_agent_model(self.agent_id) if self.agent_id else "").strip()
        installed = self._ollama_installed()

        # ---- Step 1: per-agent model is the absolute basis -----------------
        if assigned:
            # 1a. Local model assigned -> use it directly.
            if assigned in installed:
                try:
                    text = self._ollama_client().generate(
                        prompt, model=assigned, temperature=temperature)
                    if text and text.strip():
                        return text, "ollama"
                except Exception as exc:  # noqa: BLE001
                    self._last_error = exc

            # 1b. A provider specifically configures this model -> it is the
            #     primary target.
            owner = next(
                (p for p in providers if str(p.get("model", "")) == assigned), None)
            if owner is not None:
                try:
                    text = self._call_provider(owner, prompt, temperature, assigned)
                    if text and text.strip():
                        return text, str(owner.get("id", "unknown"))
                except Exception as exc:  # noqa: BLE001
                    self._last_error = exc
                providers = [p for p in providers if p is not owner]

            # 1c. Otherwise try the agent model against the rest of the chain.
            for p in providers:
                try:
                    text = self._call_provider(p, prompt, temperature, assigned)
                    if text and text.strip():
                        return text, str(p.get("id", "unknown"))
                except Exception as exc:  # noqa: BLE001
                    self._last_error = exc

        # ---- Step 2: provider chain, each with its OWN configured model -----
        for p in providers:
            try:
                text = self._call_provider(p, prompt, temperature,
                                           str(p.get("model") or ""))
                if text and text.strip():
                    return text, str(p.get("id", "unknown"))
            except Exception as exc:  # noqa: BLE001
                self._last_error = exc

        # ---- Step 3: local Ollama fallback with an INSTALLED model ----------
        try:
            model = self._ollama_fallback_model()
            text = self._ollama_client().generate(
                prompt, model=model, temperature=temperature)
            if text and text.strip():
                return text, "ollama"
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
            if self._last_error is not None:
                detail += f" | previous: {self._last_error}"
            raise AllProvidersFailed(
                f"ALL_VALIDATION_PROVIDERS_FAILED -> PAUSED_LLM_LIMIT "
                f"(last error: {detail})"
            ) from exc

        raise AllProvidersFailed(
            f"ALL_VALIDATION_PROVIDERS_FAILED -> PAUSED_LLM_LIMIT "
            f"(last error: {self._last_error})"
        )

    def _call_provider(self, provider: dict, prompt: str, temperature: float,
                       model: str) -> str:
        from openai import OpenAI

        key = self._provider_key(provider)
        if not key:
            raise RuntimeError(f"no API key stored for provider '{provider.get('name')}'")
        client = OpenAI(base_url=provider["base_url"], api_key=key)
        chat = client.chat.completions.create(
            model=model or provider.get("model", ""),
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