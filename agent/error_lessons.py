"""Error lessons — lightweight Reflexion-style memory of past tool failures.

Pure-Python, no extra LLM calls. The idea is the cheapest possible
implementation of "remember what went wrong so it doesn't happen again":

  1. When a tool returns ``{"success": false, "error": "..."}`` we
     record (tool_name, error_signature, user_query_snippet) to a small
     JSONL log at ``$HERMES_HOME/error_lessons.jsonl``.

  2. At the start of the next turn, before _build_system_prompt runs,
     we look up lessons whose ``user_query_snippet`` shares enough
     tokens with the current user message and surface them as a hint
     block. The hint is appended to the system prompt only when at
     least one lesson matches, so quiet/idle conversations are
     unaffected.

This is intentionally NOT a full Reflexion (Shinn et al. 2023) implementation:
no review LLM call, no rewriting, no critique. It's a memo system that
turns "I tried X and it failed because Y" into a system-prompt nudge for
the next similar prompt. The cost is one file read on turn start and
one append on tool error — both microseconds.

If even this is too noisy, callers can disable it via
``agent.error_lessons.enabled: false`` in config.yaml.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from typing import List

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# How many recent lessons to keep (rolling window). Older entries get
# truncated when the file grows past this — same idea as a journal.
_MAX_LESSONS = 200

# Minimum token overlap for a lesson to be considered "relevant" to a
# new query. Tuned conservatively so unrelated turns don't drag random
# lessons into the prompt.
_MIN_TOKEN_OVERLAP = 2

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _tokens(text: str) -> set:
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3}


def _lessons_path() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home() / "error_lessons.jsonl")


def _load_all() -> List[dict]:
    path = _lessons_path()
    if not os.path.exists(path):
        return []
    out: List[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.debug("error_lessons load failed: %s", exc)
    return out


def _save_all_atomic(lessons: List[dict]) -> None:
    path = _lessons_path()
    parent = os.path.dirname(path)
    try:
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".error_lessons.", suffix=".tmp", dir=parent)
        try:
            with os.fdopen(fd, "w") as f:
                for entry in lessons:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.debug("error_lessons save failed: %s", exc)


def record_error(
    tool_name: str,
    error_message: str,
    user_query: str,
    tool_args: str | None = None,
) -> None:
    """Append a single error lesson. Best-effort; never raises."""
    if not tool_name or not error_message:
        return
    entry = {
        "ts": time.time(),  # float seconds for sub-second tie-breaking
        "tool": tool_name,
        "error": (error_message or "")[:300],
        "query": (user_query or "")[:300],
        "args": (tool_args or "")[:200] if tool_args else "",
    }
    with _LOCK:
        lessons = _load_all()
        lessons.append(entry)
        # Roll the window
        if len(lessons) > _MAX_LESSONS:
            lessons = lessons[-_MAX_LESSONS:]
        _save_all_atomic(lessons)


def lessons_for_query(
    user_query: str,
    *,
    max_results: int = 3,
    min_overlap: int | None = None,
) -> List[dict]:
    """Return the most relevant lessons for the given user query.

    Relevance is token overlap on the query text. Recent lessons win
    ties. Empty list if no overlap meets ``min_overlap`` or the file
    doesn't exist.
    """
    if not user_query:
        return []
    threshold = min_overlap if min_overlap is not None else _MIN_TOKEN_OVERLAP
    query_toks = _tokens(user_query)
    if not query_toks:
        return []
    with _LOCK:
        lessons = _load_all()
    scored: List[tuple] = []
    for lesson in lessons:
        lesson_toks = _tokens(lesson.get("query", ""))
        overlap = len(query_toks & lesson_toks)
        if overlap >= threshold:
            scored.append((overlap, lesson.get("ts", 0), lesson))
    # Recent and high-overlap first
    scored.sort(key=lambda kv: (-kv[0], -kv[1]))
    return [s[2] for s in scored[:max_results]]


def format_lessons_block(lessons: List[dict]) -> str:
    """Render a small system-prompt section. Empty string if no lessons."""
    if not lessons:
        return ""
    lines = ["## Past tool errors on similar requests"]
    lines.append("Avoid repeating the failures below when you choose tools this turn.")
    for i, lesson in enumerate(lessons, 1):
        tool = lesson.get("tool", "?")
        err = lesson.get("error", "?")
        # Trim noisy stack traces / nested JSON
        err_short = err.split("\n")[0][:200]
        lines.append(f"{i}. {tool} failed: {err_short}")
    return "\n".join(lines)


def clear_all() -> None:
    """Wipe the lessons log. Used by tests and `hermes` cleanup commands."""
    path = _lessons_path()
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
