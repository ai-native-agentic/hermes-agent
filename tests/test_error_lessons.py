"""Tests for the lightweight Reflexion-style error lesson recorder.

The lessons live at $HERMES_HOME/error_lessons.jsonl and feed
build_skills_system_prompt with a small "past failures" hint when the
new user query shares enough tokens with a recorded failure.
"""
from __future__ import annotations

import json
import pytest


@pytest.fixture
def lessons_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    import agent.error_lessons
    importlib.reload(agent.error_lessons)
    yield tmp_path
    importlib.reload(agent.error_lessons)


def test_record_creates_file(lessons_home):
    from agent.error_lessons import record_error, _lessons_path
    record_error(
        tool_name="terminal",
        error_message="bash: command not found: ohmygod",
        user_query="run ohmygod",
    )
    assert (lessons_home / "error_lessons.jsonl").exists()
    with open(_lessons_path()) as f:
        line = f.readline()
    entry = json.loads(line)
    assert entry["tool"] == "terminal"
    assert "ohmygod" in entry["error"]
    assert entry["query"] == "run ohmygod"


def test_record_skips_empty_inputs(lessons_home):
    from agent.error_lessons import record_error, _load_all
    record_error(tool_name="", error_message="boom", user_query="x")
    record_error(tool_name="t", error_message="", user_query="x")
    assert _load_all() == []


def test_record_truncates_long_strings(lessons_home):
    from agent.error_lessons import record_error, _load_all
    record_error(
        tool_name="terminal",
        error_message="x" * 5000,
        user_query="y" * 5000,
    )
    e = _load_all()[0]
    assert len(e["error"]) <= 300
    assert len(e["query"]) <= 300


def test_lessons_for_query_finds_overlap(lessons_home):
    from agent.error_lessons import record_error, lessons_for_query
    record_error(
        tool_name="terminal",
        error_message="permission denied",
        user_query="install nginx package",
    )
    record_error(
        tool_name="execute_code",
        error_message="ModuleNotFoundError: pandas",
        user_query="load csv with pandas",
    )

    out = lessons_for_query("install postgresql package")
    assert len(out) == 1
    assert out[0]["tool"] == "terminal"


def test_lessons_for_query_returns_empty_when_no_overlap(lessons_home):
    from agent.error_lessons import record_error, lessons_for_query
    record_error(
        tool_name="terminal",
        error_message="oops",
        user_query="install nginx package",
    )
    out = lessons_for_query("write me a haiku about rain")
    assert out == []


def test_lessons_for_query_recent_first(lessons_home):
    """Two lessons with equal token overlap → most recent ts wins."""
    import time
    from agent.error_lessons import record_error, lessons_for_query, _save_all_atomic, _load_all
    record_error("terminal", "old error", "compile rust binary")
    time.sleep(0.01)
    record_error("terminal", "new error", "compile rust binary")
    out = lessons_for_query("compile rust binary today", max_results=2)
    assert len(out) == 2
    assert "new" in out[0]["error"]


def test_format_lessons_block_renders(lessons_home):
    from agent.error_lessons import format_lessons_block
    out = format_lessons_block([
        {"tool": "terminal", "error": "permission denied"},
        {"tool": "execute_code", "error": "ModuleNotFoundError: pandas"},
    ])
    assert "Past tool errors" in out
    assert "terminal" in out
    assert "permission denied" in out
    assert "ModuleNotFoundError" in out


def test_format_lessons_block_empty_returns_blank():
    from agent.error_lessons import format_lessons_block
    assert format_lessons_block([]) == ""


def test_rolling_window_caps_size(lessons_home):
    from agent.error_lessons import record_error, _load_all, _MAX_LESSONS
    for i in range(_MAX_LESSONS + 50):
        record_error("t", f"err {i}", f"query {i}")
    lessons = _load_all()
    assert len(lessons) == _MAX_LESSONS
    # Most recent kept (last entry should be the highest i)
    last_err = lessons[-1]["error"]
    assert str(_MAX_LESSONS + 49) in last_err


def test_clear_all_removes_file(lessons_home):
    from agent.error_lessons import record_error, clear_all, _lessons_path
    import os as _os
    record_error("t", "e", "q")
    assert _os.path.exists(_lessons_path())
    clear_all()
    assert not _os.path.exists(_lessons_path())
