import os
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="meruem_test_")
os.environ["MERUEM_DATABASE_URL"] = f"sqlite:///{os.path.join(_tmp, 'test.db')}"
# The suite must never probe the network: disable the launch-time Ollama detector.
os.environ["MERUEM_OLLAMA_DETECT_ON_LAUNCH"] = "false"


@pytest.fixture(autouse=True)
def _clean_tables():
    from app.core.db import engine, init_db
    from sqlalchemy import inspect, text

    init_db()
    with engine.begin() as conn:
        for table in inspect(engine).get_table_names():
            conn.execute(text(f"DELETE FROM {table}"))
    yield


@pytest.fixture(autouse=True)
def _force_llm_offline(monkeypatch):
    """Keep the suite deterministic: never hit a real model (cloud or local).

    Tests are written for the offline design (fast heuristic/heuristic QA path).
    On machines with a reachable Ollama the chain would otherwise run real
    inference, making timings provider-dependent. Force the validator chain to
    behave as if every provider is down; individual tests override ``generate``
    to inject fake LLM output where the LLM path is what they exercise.
    """
    from app.agents.universal_llm import AllProvidersFailed
    from app.agents.validator import ValidatorChain

    def _offline(self, prompt, temperature=0.1):
        raise AllProvidersFailed("offline test — LLM chain disabled")

    monkeypatch.setattr(ValidatorChain, "generate", _offline)


@pytest.fixture
def session():
    from app.core.db import engine, init_db
    from sqlmodel import Session

    init_db()
    with Session(engine) as s:
        yield s