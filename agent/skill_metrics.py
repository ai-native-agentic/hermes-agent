"""Lightweight per-skill usage metrics — invoke / success / last_used.

Stored as a single JSON file at ``$HERMES_HOME/skills/.usage.json``::

    {
      "csv-quick-stats": {
        "views": 12,           # skill_view calls
        "writes": 1,           # successful skill_manage create/edit/patch
        "last_used": 1775568002,  # epoch seconds
        "first_used": 1775481601
      },
      ...
    }

Concurrent-safe via a per-process Lock; the file write is atomic
(write-and-rename) so two processes can't truncate each other.

Failure modes are intentionally silent: metrics are best-effort and
must never block tool execution. Any IO error is logged at DEBUG and
the operation returns without raising.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def _metrics_path() -> str:
    """Resolve $HERMES_HOME/skills/.usage.json (does not create dirs)."""
    from hermes_constants import get_hermes_home
    return str(get_hermes_home() / "skills" / ".usage.json")


def _load() -> Dict[str, Dict[str, Any]]:
    path = _metrics_path()
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}


def _save_atomic(data: Dict[str, Dict[str, Any]]) -> None:
    path = _metrics_path()
    parent = os.path.dirname(path)
    try:
        os.makedirs(parent, exist_ok=True)
        # Atomic write — write to a temp file in the same dir then rename.
        fd, tmp = tempfile.mkstemp(prefix=".usage.", suffix=".tmp", dir=parent)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.debug("skill metrics save failed: %s", exc)


def record_view(skill_name: str) -> None:
    """Increment the view counter for ``skill_name`` (best-effort)."""
    if not skill_name:
        return
    now = int(time.time())
    with _LOCK:
        data = _load()
        entry = data.setdefault(skill_name, {})
        entry["views"] = int(entry.get("views", 0)) + 1
        entry["last_used"] = now
        entry.setdefault("first_used", now)
        _save_atomic(data)


def record_write(skill_name: str) -> None:
    """Increment the write counter (create / edit / patch successes)."""
    if not skill_name:
        return
    now = int(time.time())
    with _LOCK:
        data = _load()
        entry = data.setdefault(skill_name, {})
        entry["writes"] = int(entry.get("writes", 0)) + 1
        entry["last_used"] = now
        entry.setdefault("first_used", now)
        _save_atomic(data)


def load_metrics() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the current metrics dict."""
    with _LOCK:
        return dict(_load())


def top_used(n: int = 5) -> list:
    """Return up to ``n`` (skill_name, views, last_used) sorted by views desc."""
    data = load_metrics()
    rows = [(name, int(e.get("views", 0)), int(e.get("last_used", 0)))
            for name, e in data.items()]
    rows.sort(key=lambda r: (-r[1], -r[2]))
    return rows[:n]


def unused_since(cutoff_epoch: int, all_skill_names: Iterable[str]) -> list:
    """Return names from ``all_skill_names`` whose last_used is < cutoff
    (or has no record at all). Used for the "unused skills" insights panel
    and the optional auto-archive flow.
    """
    data = load_metrics()
    out = []
    for name in all_skill_names:
        entry = data.get(name)
        if not entry:
            out.append(name)
            continue
        if int(entry.get("last_used", 0)) < cutoff_epoch:
            out.append(name)
    return sorted(out)


def archive_skill(skill_basename: str) -> dict:
    """Move a single user/agent-created skill into the archive bucket.

    The archive lives at ``$HERMES_HOME/skills/.archive/<category>/<name>/``.
    Bundled skills (anything in ``.bundled_manifest``) are NEVER archived
    so this is safe to call from automation. Returns a small status dict
    with ``moved_from`` / ``moved_to`` (or ``error`` on failure).

    Side effect: clears the in-process skills prompt cache so the next
    turn rebuilds without the archived skill.
    """
    import shutil
    from pathlib import Path
    from hermes_constants import get_hermes_home

    skills_root = get_hermes_home() / "skills"
    archive_root = skills_root / ".archive"

    # Refuse to archive bundled skills
    bundled = set()
    manifest = skills_root / ".bundled_manifest"
    if manifest.exists():
        try:
            with open(manifest) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        bundled.add(line.split(":", 1)[0])
        except OSError:
            pass
    if skill_basename in bundled:
        return {"error": f"refusing to archive bundled skill {skill_basename!r}"}

    # Find the skill on disk
    candidates = []
    if skills_root.exists():
        for sm in skills_root.glob("*/*/SKILL.md"):
            if sm.parent.name == skill_basename:
                candidates.append(sm.parent)
    if not candidates:
        return {"error": f"skill {skill_basename!r} not found under {skills_root}"}
    src = candidates[0]
    rel = src.relative_to(skills_root)  # e.g. "writing/hello-haiku"
    dst = archive_root / rel
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
    except OSError as exc:
        return {"error": str(exc)}

    # Drop usage record so insights stops listing the archived skill
    with _LOCK:
        data = _load()
        if skill_basename in data:
            data.pop(skill_basename, None)
            _save_atomic(data)

    # Bust the prompt builder's cache so the next turn rebuilds the index
    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass

    return {"moved_from": str(src), "moved_to": str(dst)}


def auto_archive_unused(days: int = 30, all_skill_names: Iterable[str] | None = None) -> dict:
    """Archive every user skill that hasn't been touched in ``days`` days.

    When ``all_skill_names`` is None, the function discovers them by
    scanning $HERMES_HOME/skills/ and skipping the bundled set. Returns
    a summary dict::

        {"archived": [name, ...], "errors": {name: msg, ...}, "scanned": N}
    """
    import time as _time
    from hermes_constants import get_hermes_home

    cutoff = int(_time.time() - days * 86400)
    skills_root = get_hermes_home() / "skills"

    if all_skill_names is None:
        all_skill_names = []
        bundled = set()
        manifest = skills_root / ".bundled_manifest"
        if manifest.exists():
            try:
                with open(manifest) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            bundled.add(line.split(":", 1)[0])
            except OSError:
                pass
        if skills_root.exists():
            for sm in skills_root.glob("*/*/SKILL.md"):
                basename = sm.parent.name
                if basename not in bundled:
                    all_skill_names.append(basename)

    candidates = unused_since(cutoff, all_skill_names)
    archived: list[str] = []
    errors: dict[str, str] = {}
    for name in candidates:
        result = archive_skill(name)
        if "error" in result:
            errors[name] = result["error"]
        else:
            archived.append(name)
    return {
        "archived": archived,
        "errors": errors,
        "scanned": len(list(all_skill_names)) if not isinstance(all_skill_names, list) else len(all_skill_names),
    }
