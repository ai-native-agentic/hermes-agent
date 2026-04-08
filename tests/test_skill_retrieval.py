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


# ── A2: usage_boost integration ─────────────────────────────────────────

@pytest.fixture
def usage_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a clean tmp dir so the in-process metrics
    helper writes/reads our isolated .usage.json instead of the real one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "skills").mkdir()
    import importlib
    import agent.skill_metrics
    importlib.reload(agent.skill_metrics)
    yield tmp_path
    importlib.reload(agent.skill_metrics)


def test_usage_boost_promotes_frequently_used_skill(usage_home):
    """Two skills tied on TF-IDF score — the one with more usage views
    should rank first when usage_boost=True."""
    from agent.skill_metrics import record_view
    skills = [
        {"name": "rare-skill", "description": "do thing alpha beta"},
        {"name": "popular-skill", "description": "do thing alpha beta"},
    ]
    for _ in range(10):
        record_view("popular-skill")

    boosted = topk_skills("alpha beta", skills, k=2, usage_boost=True)
    assert boosted[0]["name"] == "popular-skill"

    plain = topk_skills("alpha beta", skills, k=2, usage_boost=False)
    # Without boost, the order is whatever score_skills returns (tie → input order)
    assert {s["name"] for s in plain} == {"rare-skill", "popular-skill"}


def test_usage_boost_preserves_query_relevance(usage_home):
    """Heavy usage should NOT pull an irrelevant skill above a clearly
    matching one — relevance dominates, boost only breaks ties."""
    from agent.skill_metrics import record_view
    skills = [
        {"name": "git-bisect", "description": "git regression bisect commits"},
        {"name": "popular-irrelevant", "description": "pizza recipe pasta"},
    ]
    for _ in range(50):
        record_view("popular-irrelevant")

    out = topk_skills("git regression", skills, k=1, usage_boost=True)
    assert out[0]["name"] == "git-bisect"


def test_usage_boost_default_is_on():
    """The wire-up is opt-out, not opt-in — keyword arg defaults True."""
    import inspect
    sig = inspect.signature(topk_skills)
    assert sig.parameters["usage_boost"].default is True


def test_usage_boost_handles_missing_metrics_gracefully(tmp_path, monkeypatch):
    """If the metrics file doesn't exist or load_metrics blows up, the
    plain TF-IDF order must still come back — no exception."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills = [
        {"name": "a", "description": "alpha"},
        {"name": "b", "description": "beta"},
    ]
    out = topk_skills("alpha", skills, k=1, usage_boost=True)
    assert out[0]["name"] == "a"


def test_usage_boost_zero_views_keeps_floor(usage_home):
    """A new skill with 0 views still gets the +1 floor multiplier so it
    isn't permanently locked out by older popular skills."""
    from agent.skill_metrics import record_view
    skills = [
        {"name": "old-popular", "description": "alpha topic"},
        {"name": "new-skill", "description": "alpha topic"},
    ]
    # old-popular has been viewed a few times
    for _ in range(3):
        record_view("old-popular")

    # The boost is log(1+3)+1 ≈ 2.39 vs 1.0 — old wins, but new isn't 0
    out = topk_skills("alpha topic", skills, k=2, usage_boost=True)
    assert "new-skill" in [s["name"] for s in out]
