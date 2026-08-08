"""Universal LLM provider chain tests (offline / no network)."""

import pytest

from app.agents.universal_llm import AllProvidersFailed, UniversalLLM
from app.agents.validator import ValidatorChain
from app.services.model_config import get_providers


def test_default_providers_have_required_fields():
    for p in get_providers():
        assert p["id"]
        assert p["name"]
        assert p["base_url"]
        assert p["model"]
        assert isinstance(p["enabled"], bool)
        assert ":" in p["key_ref"]


def test_defaults_contain_no_deepseek_or_openai():
    ids = {p["id"] for p in get_providers()}
    assert "deepseek" not in ids
    assert "openai" not in ids


def test_generate_raises_when_all_providers_fail(monkeypatch):
    def fake_providers():
        return [
            {"id": "broken", "name": "Broken", "base_url": "http://127.0.0.1:9",
             "model": "x", "enabled": True, "key_ref": "llm:broken"},
        ]

    monkeypatch.setattr("app.agents.universal_llm.get_providers", fake_providers)
    monkeypatch.setattr(
        "app.agents.universal_llm.UniversalLLM._provider_key", lambda self, p: "dummy-key")
    with pytest.raises(AllProvidersFailed):
        UniversalLLM().generate("hello", temperature=0.0)


def test_chain_validate_surfaces_validation_failed(monkeypatch):
    """When everything is down, validate() raises ValidationFailed (not raw)."""
    import app.agents.validator as validator_mod
    from app.agents.universal_llm import AllProvidersFailed

    def fake_generate(self, prompt):
        raise AllProvidersFailed("all down")

    monkeypatch.setattr(validator_mod.UniversalLLM, "generate", fake_generate)
    chain = ValidatorChain()
    with pytest.raises(validator_mod.ValidationFailed):
        chain.validate({"asset": "XAUUSD"}, {"market": "ctx"})
