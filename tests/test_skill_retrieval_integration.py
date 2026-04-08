"""Integration tests for the build_skills_system_prompt → retrieval wiring.

Validates:
  - Default config (no skills.retrieval section) → unchanged "inject all" path
  - mode="topk" with a relevant query → only top-k skills appear
  - mode="topk" with an irrelevant query + min_score → may filter to empty
  - mode="topk" with no query → falls through to first-k by insertion order
  - Cache key includes the query, so different queries give different results
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agent.prompt_builder import build_skills_system_prompt, _SKILLS_PROMPT_CACHE


@pytest.fixture
def skills_workspace(tmp_path, monkeypatch):
    """Build a tiny ~/.hermes/skills tree with 5 distinct skills."""
    home = tmp_path / "hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)

    sample = [
        ("devops", "git-bisect-walkthrough", "Find a git regression by bisecting commits"),
        ("writing", "csv-quick-stats", "Load a CSV with pandas and report mean min max"),
        ("devops", "docker-cleanup", "Remove stopped containers and dangling docker images"),
        ("writing", "haiku-generator", "Compose a 5-7-5 syllable haiku about a topic"),
        ("writing", "python-snippet-runner", "Run a python snippet and capture stdout"),
    ]
    for cat, name, desc in sample:
        d = skills / cat / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n\nbody\n")

    monkeypatch.setenv("HERMES_HOME", str(home))
    # Drop the in-process LRU so tmp_path snapshots aren't masked by an
    # earlier test using the real ~/.hermes location.
    _SKILLS_PROMPT_CACHE.clear()
    yield home
    _SKILLS_PROMPT_CACHE.clear()


def test_default_config_includes_all_skills(skills_workspace):
    """No retrieval config → all 5 skills must appear in the prompt."""
    out = build_skills_system_prompt()
    assert "git-bisect-walkthrough" in out
    assert "csv-quick-stats" in out
    assert "docker-cleanup" in out
    assert "haiku-generator" in out
    assert "python-snippet-runner" in out


def test_topk_filters_to_query_match(skills_workspace):
    """A git-bisect query should yield only the bisect skill at top-k=1."""
    _SKILLS_PROMPT_CACHE.clear()
    out = build_skills_system_prompt(
        user_query="find a regression in my git history",
        retrieval_config={"mode": "topk", "top_k": 1},
    )
    assert "git-bisect-walkthrough" in out
    assert "csv-quick-stats" not in out
    assert "haiku-generator" not in out


def test_topk_returns_multiple_when_k_higher(skills_workspace):
    _SKILLS_PROMPT_CACHE.clear()
    out = build_skills_system_prompt(
        user_query="csv pandas analytics",
        retrieval_config={"mode": "topk", "top_k": 2},
    )
    assert "csv-quick-stats" in out
    # The python-snippet-runner is plausible second match
    other_count = sum(
        1 for n in [
            "git-bisect-walkthrough",
            "docker-cleanup",
            "haiku-generator",
            "python-snippet-runner",
        ] if n in out
    )
    # k=2 means: csv-quick-stats + 1 other
    assert other_count == 1


def test_topk_falls_back_to_all_when_query_missing(skills_workspace):
    """No user_query → retrieval is inactive, prompt is unfiltered."""
    _SKILLS_PROMPT_CACHE.clear()
    out = build_skills_system_prompt(
        user_query=None,
        retrieval_config={"mode": "topk", "top_k": 1},
    )
    assert "git-bisect-walkthrough" in out
    assert "csv-quick-stats" in out
    assert "docker-cleanup" in out


def test_topk_with_min_score_drops_irrelevant_query(skills_workspace):
    """Garbage query + min_score → no skills clear the bar → empty prompt."""
    _SKILLS_PROMPT_CACHE.clear()
    out = build_skills_system_prompt(
        user_query="qzx blarg flibbertigibbet",
        retrieval_config={"mode": "topk", "top_k": 3, "min_score": 0.5},
    )
    assert out == ""


def test_topk_cache_distinguishes_queries(skills_workspace):
    """Different queries must NOT collide in the LRU cache."""
    _SKILLS_PROMPT_CACHE.clear()
    out_git = build_skills_system_prompt(
        user_query="git regression bisect",
        retrieval_config={"mode": "topk", "top_k": 1},
    )
    out_haiku = build_skills_system_prompt(
        user_query="compose a haiku about syllable patterns",
        retrieval_config={"mode": "topk", "top_k": 1},
    )
    assert "git-bisect" in out_git
    assert "haiku" in out_haiku
    assert out_git != out_haiku


def test_mode_all_explicit_is_same_as_default(skills_workspace):
    _SKILLS_PROMPT_CACHE.clear()
    out_default = build_skills_system_prompt()
    _SKILLS_PROMPT_CACHE.clear()
    out_explicit = build_skills_system_prompt(
        user_query="anything",
        retrieval_config={"mode": "all"},
    )
    # Both should contain all 5
    for skill in ("git-bisect", "csv-quick-stats", "docker-cleanup", "haiku", "python-snippet"):
        assert skill in out_default
        assert skill in out_explicit
