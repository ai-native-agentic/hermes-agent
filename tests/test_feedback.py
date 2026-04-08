"""Tests for the user feedback (/rate) channel."""
from __future__ import annotations

import json
import pytest


@pytest.fixture
def feedback_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib, agent.feedback
    importlib.reload(agent.feedback)
    yield tmp_path
    importlib.reload(agent.feedback)


def test_record_thumbs_up(feedback_home):
    from agent.feedback import record_rating, load_all
    assert record_rating("up", session_id="s1", model="Qwen3-32B") is True
    entries = load_all()
    assert len(entries) == 1
    assert entries[0]["rating"] == "up"
    assert entries[0]["session_id"] == "s1"
    assert entries[0]["model"] == "Qwen3-32B"


def test_record_thumbs_down_with_reason(feedback_home):
    from agent.feedback import record_rating, load_all
    record_rating("down", reason="wrong answer")
    e = load_all()[0]
    assert e["rating"] == "down"
    assert e["reason"] == "wrong answer"


def test_record_normalizes_aliases(feedback_home):
    from agent.feedback import record_rating, load_all
    assert record_rating("+") is True
    assert record_rating("-") is True
    assert record_rating("👍") is True
    assert record_rating("👎") is True
    assert record_rating("y") is True
    assert record_rating("n") is True
    ratings = [e["rating"] for e in load_all()]
    assert ratings == ["up", "down", "up", "down", "up", "down"]


def test_record_rejects_garbage_rating(feedback_home):
    from agent.feedback import record_rating, load_all
    assert record_rating("maybe") is False
    assert record_rating("") is False
    assert record_rating(None) is False  # type: ignore[arg-type]
    assert load_all() == []


def test_stats_computes_ratio(feedback_home):
    from agent.feedback import record_rating, stats
    for _ in range(7):
        record_rating("up")
    for _ in range(3):
        record_rating("down")
    s = stats(days=30)
    assert s["up"] == 7
    assert s["down"] == 3
    assert s["total"] == 10
    assert s["ratio"] == 0.7


def test_stats_zero_when_empty(feedback_home):
    from agent.feedback import stats
    s = stats()
    assert s["total"] == 0
    assert s["ratio"] == 0.0


def test_record_truncates_long_strings(feedback_home):
    from agent.feedback import record_rating, load_all
    record_rating(
        "up",
        reason="x" * 5000,
        user_message="y" * 5000,
        assistant_response="z" * 5000,
    )
    e = load_all()[0]
    assert len(e["reason"]) <= 500
    assert len(e["user_message"]) <= 500
    assert len(e["assistant_response"]) <= 1000
