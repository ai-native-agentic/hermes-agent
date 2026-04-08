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
