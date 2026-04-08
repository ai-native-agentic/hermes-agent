"""Tests for the per-model MoA metrics + weight derivation helper."""
from __future__ import annotations

import pytest


@pytest.fixture
def metrics_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib, agent.moa_metrics
    importlib.reload(agent.moa_metrics)
    yield tmp_path
    importlib.reload(agent.moa_metrics)


def test_record_call_creates_entry(metrics_home):
    from agent.moa_metrics import record_call, load_metrics
    record_call("Qwen3-32B", success=True)
    data = load_metrics()
    assert data["Qwen3-32B"]["calls"] == 1
    assert data["Qwen3-32B"]["successes"] == 1
    assert "last_call" in data["Qwen3-32B"]


def test_record_failure_does_not_bump_successes(metrics_home):
    from agent.moa_metrics import record_call, load_metrics
    record_call("flaky-model", success=False)
    record_call("flaky-model", success=False)
    record_call("flaky-model", success=True)
    data = load_metrics()
    assert data["flaky-model"]["calls"] == 3
    assert data["flaky-model"]["successes"] == 1


def test_record_call_ignores_empty_model(metrics_home):
    from agent.moa_metrics import record_call, load_metrics
    record_call("", success=True)
    record_call(None, success=False)  # type: ignore[arg-type]
    assert load_metrics() == {}


def test_derive_weights_uniform_when_no_data(metrics_home):
    from agent.moa_metrics import derive_weights
    weights = derive_weights(["a", "b", "c"])
    assert pytest.approx(sum(weights.values())) == 1.0
    assert weights["a"] == weights["b"] == weights["c"] == pytest.approx(1.0 / 3)


def test_derive_weights_favors_higher_success_rate(metrics_home):
    from agent.moa_metrics import record_call, derive_weights
    # Model A: 10 calls, all success
    for _ in range(10):
        record_call("good", success=True)
    # Model B: 10 calls, 5 success
    for i in range(10):
        record_call("mid", success=(i % 2 == 0))
    # Model C: 10 calls, 1 success
    for i in range(10):
        record_call("bad", success=(i == 0))
    weights = derive_weights(["good", "mid", "bad"])
    assert pytest.approx(sum(weights.values()), abs=1e-6) == 1.0
    assert weights["good"] > weights["mid"] > weights["bad"]


def test_derive_weights_favors_more_data(metrics_home):
    """Two models with the same 100% success rate but different sample
    sizes — log smoothing means the 100-call model gets more weight."""
    from agent.moa_metrics import record_call, derive_weights
    for _ in range(2):
        record_call("new", success=True)
    for _ in range(100):
        record_call("seasoned", success=True)
    weights = derive_weights(["new", "seasoned"])
    assert weights["seasoned"] > weights["new"]


def test_derive_weights_floors_low_scores(metrics_home):
    """A bad model should never drop below the floor (10% of max by default)
    so we keep some diversity in the weighted vote."""
    from agent.moa_metrics import record_call, derive_weights
    for _ in range(50):
        record_call("good", success=True)
    record_call("terrible", success=False)
    weights = derive_weights(["good", "terrible"])
    # terrible has weight=0 raw → after floor it gets 10% of good
    assert weights["terrible"] > 0
    assert weights["good"] > weights["terrible"]
    assert pytest.approx(sum(weights.values()), abs=1e-6) == 1.0


def test_derive_weights_handles_model_not_in_metrics(metrics_home):
    from agent.moa_metrics import record_call, derive_weights
    record_call("known", success=True)
    weights = derive_weights(["known", "stranger"])
    assert "known" in weights and "stranger" in weights
    assert pytest.approx(sum(weights.values()), abs=1e-6) == 1.0
