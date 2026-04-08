"""Tests for the dependency-free TF-IDF skill retrieval helper."""
from __future__ import annotations

import pytest

from agent.skill_retrieval import score_skills, topk_skills, _tokenize


# ── Tokenizer sanity ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Hello world", ["hello", "world"]),
    ("Run pytest with -v flag!", ["run", "pytest", "with", "flag"]),
    ("CamelCase keeps the words", ["camelcase", "keeps", "the", "words"]),
    ("", []),
    ("a I U", []),  # all tokens shorter than 2 chars dropped
    ("git-bisect-walkthrough", ["git", "bisect", "walkthrough"]),
])
def test_tokenize(text, expected):
    assert _tokenize(text) == expected


# ── Ranking semantics ────────────────────────────────────────────────────

SKILLS = [
    {"name": "git-bisect-walkthrough",
     "description": "Find a git regression by bisecting commits"},
    {"name": "csv-quick-stats",
     "description": "Load a CSV with pandas and report mean min max"},
    {"name": "docker-cleanup",
     "description": "Remove stopped containers and dangling images"},
    {"name": "haiku-generator",
     "description": "Compose a 5-7-5 syllable poem about a topic"},
    {"name": "python-snippet-runner",
     "description": "Safely run a python snippet and capture stdout"},
]


def test_query_finds_most_relevant_skill_first():
    scored = score_skills("how do I find a regression in my git history?", SKILLS)
    top = scored[0][0]
    assert top["name"] == "git-bisect-walkthrough"


def test_csv_query_picks_csv_skill():
    scored = score_skills("load a csv and tell me the mean", SKILLS)
    assert scored[0][0]["name"] == "csv-quick-stats"


def test_docker_query_picks_docker_skill():
    scored = score_skills("clean up dangling docker images", SKILLS)
    assert scored[0][0]["name"] == "docker-cleanup"


def test_haiku_query_picks_haiku_skill():
    scored = score_skills("write me a poem", SKILLS)
    assert scored[0][0]["name"] == "haiku-generator"


def test_unrelated_query_returns_low_or_zero_scores():
    scored = score_skills("what time is it in tokyo?", SKILLS)
    # No skill should be obviously relevant — top score should be small
    assert scored[0][1] < 0.3


# ── topk_skills wrapper ──────────────────────────────────────────────────

def test_topk_returns_at_most_k_skills():
    out = topk_skills("git regression", SKILLS, k=2)
    assert len(out) == 2
    assert out[0]["name"] == "git-bisect-walkthrough"


def test_topk_with_min_score_filters_out_irrelevant():
    out = topk_skills("alkdsfjlkasdjf", SKILLS, k=5, min_score=0.1)
    assert out == []


def test_topk_returns_empty_when_k_is_zero():
    assert topk_skills("anything", SKILLS, k=0) == []


def test_topk_empty_skills_returns_empty():
    assert topk_skills("anything", [], k=5) == []


def test_topk_empty_query_returns_first_k_unfiltered():
    """When the query is empty we can't rank — fall back to insertion order
    so callers asking for top-k always get a slice of the available skills."""
    out = topk_skills("", SKILLS, k=3)
    assert len(out) == 3
    assert [s["name"] for s in out] == [
        "git-bisect-walkthrough",
        "csv-quick-stats",
        "docker-cleanup",
    ]


def test_score_skills_handles_missing_fields_gracefully():
    weird_skills = [
        {"name": "only-name"},
        {"description": "only description here"},
        {},  # nothing at all
        {"name": "csv-thing", "description": "csv csv csv"},
    ]
    scored = score_skills("csv", weird_skills)
    assert scored[0][0]["name"] == "csv-thing"


def test_score_skills_idf_downweights_common_terms():
    """Words that appear in every skill should NOT dominate the ranking."""
    skills = [
        {"name": "a", "description": "skill skill skill alpha"},
        {"name": "b", "description": "skill skill skill beta"},
        {"name": "c", "description": "skill skill skill gamma"},
    ]
    # 'skill' is in every doc → IDF ≈ 1.0 (still positive but flat)
    # 'alpha' is unique → high IDF
    scored = score_skills("alpha", skills)
    assert scored[0][0]["name"] == "a"
