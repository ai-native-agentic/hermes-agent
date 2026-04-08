"""Tests for the per-skill usage metrics helper.

The metrics file lives at $HERMES_HOME/skills/.usage.json. We point
HERMES_HOME at a tmp_path so the tests don't touch the real one.
"""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def metrics_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "skills").mkdir()
    # Reset module-level lock + reload to drop any cached state
    import importlib
    import agent.skill_metrics
    importlib.reload(agent.skill_metrics)
    yield tmp_path
    importlib.reload(agent.skill_metrics)


def test_record_view_creates_entry(metrics_home):
    from agent.skill_metrics import record_view, load_metrics
    record_view("csv-quick-stats")
    data = load_metrics()
    assert "csv-quick-stats" in data
    assert data["csv-quick-stats"]["views"] == 1
    assert "last_used" in data["csv-quick-stats"]
    assert "first_used" in data["csv-quick-stats"]


def test_record_view_increments(metrics_home):
    from agent.skill_metrics import record_view, load_metrics
    for _ in range(3):
        record_view("csv-quick-stats")
    assert load_metrics()["csv-quick-stats"]["views"] == 3


def test_record_write_separate_counter(metrics_home):
    from agent.skill_metrics import record_view, record_write, load_metrics
    record_view("foo")
    record_write("foo")
    record_write("foo")
    data = load_metrics()["foo"]
    assert data["views"] == 1
    assert data["writes"] == 2


def test_record_view_ignores_empty_name(metrics_home):
    from agent.skill_metrics import record_view, load_metrics
    record_view("")
    record_view(None)  # type: ignore[arg-type]
    assert load_metrics() == {}


def test_top_used_orders_by_view_count(metrics_home):
    from agent.skill_metrics import record_view, top_used
    for _ in range(5):
        record_view("csv-quick-stats")
    for _ in range(2):
        record_view("haiku-generator")
    record_view("docker-cleanup")
    out = top_used(n=3)
    assert [r[0] for r in out] == ["csv-quick-stats", "haiku-generator", "docker-cleanup"]
    assert [r[1] for r in out] == [5, 2, 1]


def test_unused_since_returns_skills_with_no_metrics(metrics_home):
    from agent.skill_metrics import record_view, unused_since
    record_view("recent")
    out = unused_since(0, ["recent", "never-touched-1", "never-touched-2"])
    assert out == ["never-touched-1", "never-touched-2"]


def test_unused_since_uses_cutoff(metrics_home):
    """Skills last used before cutoff are returned as unused."""
    from agent.skill_metrics import record_view, load_metrics, _save_atomic, _load
    # Stamp an old skill manually
    record_view("ancient")
    record_view("recent")
    data = _load()
    data["ancient"]["last_used"] = 1
    _save_atomic(data)

    out = unused_since(time.time() - 10, ["ancient", "recent"])
    assert out == ["ancient"]


def test_metrics_file_is_valid_json_after_writes(metrics_home):
    from agent.skill_metrics import record_view, _metrics_path
    for name in ["a", "b", "c"]:
        record_view(name)
        record_view(name)
    with open(_metrics_path()) as f:
        data = json.load(f)
    assert set(data.keys()) == {"a", "b", "c"}
    for entry in data.values():
        assert entry["views"] == 2


def test_atomic_write_does_not_corrupt_on_concurrent_call(metrics_home):
    """Hammer the recorder from a few threads and verify the file
    remains parseable JSON throughout."""
    import threading
    from agent.skill_metrics import record_view, _metrics_path

    def hammer(name, n=20):
        for _ in range(n):
            record_view(name)

    threads = [threading.Thread(target=hammer, args=(f"s{i}",)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    with open(_metrics_path()) as f:
        data = json.load(f)  # must parse cleanly
    for i in range(4):
        assert data[f"s{i}"]["views"] == 20


def _import_unused_since():
    from agent.skill_metrics import unused_since
    return unused_since


# late-bound to keep ruff/pylint happy in test_unused_since_uses_cutoff
unused_since = _import_unused_since()


# ── Archive helpers ──────────────────────────────────────────────────────

def _seed_skill(home, category, name, body="body"):
    """Create a SKILL.md under home/skills/<category>/<name>/."""
    d = home / "skills" / category / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: test\n---\n\n{body}")
    return d


def test_archive_skill_moves_to_archive_dir(metrics_home):
    from agent.skill_metrics import archive_skill, record_view, _metrics_path
    _seed_skill(metrics_home, "writing", "haiku-test")
    record_view("haiku-test")  # add a usage record
    result = archive_skill("haiku-test")
    assert "moved_to" in result
    assert (metrics_home / "skills" / ".archive" / "writing" / "haiku-test" / "SKILL.md").exists()
    assert not (metrics_home / "skills" / "writing" / "haiku-test").exists()
    # Usage record should be wiped after archive
    import json
    data = json.loads((metrics_home / "skills" / ".usage.json").read_text())
    assert "haiku-test" not in data


def test_archive_refuses_bundled_skills(metrics_home):
    from agent.skill_metrics import archive_skill
    # Build a fake bundled manifest that lists "github-pr-workflow"
    (metrics_home / "skills" / ".bundled_manifest").write_text("github-pr-workflow:abc123\n")
    _seed_skill(metrics_home, "github", "github-pr-workflow")
    result = archive_skill("github-pr-workflow")
    assert "error" in result
    assert "bundled" in result["error"].lower()
    # File must still be present
    assert (metrics_home / "skills" / "github" / "github-pr-workflow" / "SKILL.md").exists()


def test_archive_unknown_skill_returns_error(metrics_home):
    from agent.skill_metrics import archive_skill
    result = archive_skill("not-a-real-skill")
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_auto_archive_unused_skips_recent(metrics_home):
    from agent.skill_metrics import auto_archive_unused, record_view
    _seed_skill(metrics_home, "writing", "fresh-skill")
    _seed_skill(metrics_home, "writing", "stale-skill")
    record_view("fresh-skill")  # marks fresh-skill as recently used

    summary = auto_archive_unused(days=30)
    assert "stale-skill" in summary["archived"]
    assert "fresh-skill" not in summary["archived"]
    assert (metrics_home / "skills" / ".archive" / "writing" / "stale-skill").exists()
    assert (metrics_home / "skills" / "writing" / "fresh-skill" / "SKILL.md").exists()


def test_auto_archive_unused_with_zero_days_archives_everything(metrics_home):
    from agent.skill_metrics import auto_archive_unused
    _seed_skill(metrics_home, "writing", "a")
    _seed_skill(metrics_home, "devops", "b")
    summary = auto_archive_unused(days=0)
    assert set(summary["archived"]) == {"a", "b"}
