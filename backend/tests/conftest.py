import os
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="meruem_test_")
os.environ["MERUEM_DATABASE_URL"] = f"sqlite:///{os.path.join(_tmp, 'test.db')}"


@pytest.fixture(autouse=True)
def _clean_tables():
    from app.core.db import engine, init_db
    from sqlalchemy import inspect, text

    init_db()
    with engine.begin() as conn:
        for table in inspect(engine).get_table_names():
            conn.execute(text(f"DELETE FROM {table}"))
    yield


@pytest.fixture
def session():
    from app.core.db import engine, init_db
    from sqlmodel import Session

    init_db()
    with Session(engine) as s:
        yield s