"""Julia compute layer tests (QiMera.md §6.4 / M2).

Skipped cleanly when the Julia runtime or juliacall is not installed — the
bridge is designed so the API degrades gracefully, and these tests mirror that.
"""

import pytest
from fastapi.testclient import TestClient

from qimera.app import julia_bridge as jb
from qimera.app.main import app

pytestmark = pytest.mark.skipif(
    not jb.available(), reason="Julia runtime / juliacall not installed"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_fitness_parity_between_julia_and_python():
    close = jb._synthetic_close(5000)
    high, low = jb._synthetic_ohlc(close)
    params = (50, 14, 3.0)

    fit_jl = jb.fitness(close, high, low, *params)
    fit_py = jb.fitness_python(close, high, low, *params)

    assert len(fit_jl) == 5
    assert all(abs(a - b) < 1e-9 for a, b in zip(fit_jl, fit_py))


def test_sweep_parity_between_julia_and_python():
    close = jb._synthetic_close(4000)
    lookback = 150

    mn_j, mx_j = jb.sweep_stats(close, lookback)
    mn_p, mx_p = jb.sweep_python(close, lookback)

    assert len(mn_j) == len(close)
    assert mn_j == pytest.approx(mn_p, abs=1e-9)
    assert mx_j == pytest.approx(mx_p, abs=1e-9)


def test_benchmark_reports_parity_and_speedup():
    res = jb.benchmark(bars=8000, lookback=100, runs=2)

    assert res["parity_fitness"] is True
    assert res["parity_sweep"] is True
    assert res["julia_sweep_s"] > 0
    assert res["python_sweep_s"] > 0
    assert res["julia_sweep_s"] <= res["python_sweep_s"]
    assert res["fitness"] is not None


def test_api_julia_status(client):
    r = client.get("/api/julia/status")

    assert r.status_code == 200
    data = r.json()
    assert data["available"] is True
    assert data["module_loaded"] is True
    assert "fitness" in (data["functions"] or [])


def test_api_benchmark_endpoint(client):
    r = client.post("/api/benchmark", json={"bars": 5000, "lookback": 80, "runs": 1})

    assert r.status_code == 200
    data = r.json()
    assert data["parity_sweep"] is True
    assert data["speedup"] is not None