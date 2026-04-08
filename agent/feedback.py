"""User feedback channel — append-only thumbs up/down log.

The /rate slash command (CLI + gateway) calls into here to record
explicit user judgement on the last assistant response. The data is
stored as JSONL at ``$HERMES_HOME/feedback.jsonl``::

    {"ts": 1775680123.4, "session_id": "20260409_...", "rating": "up",
     "reason": "", "model": "Qwen3-32B", "user_message": "...",
     "assistant_response": "..."}

This is the first explicit user-side feedback channel in Hermes.
Previously the agent had to infer satisfaction from /retry / /undo
patterns, which is noisy. Concrete uses for the data once it's flowing:

  1. ``hermes insights`` thumbs ratio (current vs trend)
  2. Future MoA aggregator weight: prefer reference models that more
     often produced the snippet the aggregator chose AND that the user
     thumbs-upped
  3. Future Reflexion-style negative example pool

The recorder is best-effort and never blocks the main loop.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def _feedback_path() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home() / "feedback.jsonl")


def record_rating(
    rating: str,
    *,
    session_id: str = "",
    reason: str = "",
    model: str = "",
    user_message: str = "",
    assistant_response: str = "",
) -> bool:
    """Append one rating entry. Returns True on success.

    Rating must be "up", "down", or anything starting with those
    (case-insensitive). Anything else is rejected with a False return
    so the caller can show usage help.
    """
    norm = (rating or "").strip().lower()
    if not norm:
        return False
    if norm[0] in ("u", "+", "👍", "y"):
        norm = "up"
    elif norm[0] in ("d", "-", "👎", "n"):
        norm = "down"
    else:
        return False

    entry = {
        "ts": time.time(),
        "session_id": (session_id or "")[:64],
        "rating": norm,
        "reason": (reason or "")[:500],
        "model": (model or "")[:64],
        "user_message": (user_message or "")[:500],
        "assistant_response": (assistant_response or "")[:1000],
    }
    path = _feedback_path()
    parent = os.path.dirname(path)
    try:
        with _LOCK:
            os.makedirs(parent, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.debug("feedback record failed: %s", exc)
        return False

    # Bridge: a thumbs-down with concrete user/assistant context turns
    # into an error lesson so the next prompt that overlaps with this
    # one can pre-emptively warn the model. This is the closed loop
    # equivalent of "the user told us this answer was bad — remember it".
    # Best-effort, never raises.
    if norm == "down" and user_message and assistant_response:
        try:
            from agent.error_lessons import record_error
            record_error(
                tool_name="(user_thumbs_down)",
                error_message=(
                    f"User rated this response negative."
                    + (f" Reason: {reason}" if reason else "")
                    + f" Response excerpt: {assistant_response[:200]}"
                ),
                user_query=user_message,
            )
        except Exception:
            pass
    return True


def load_all() -> list:
    """Return all recorded feedback entries (most recent at end)."""
    path = _feedback_path()
    if not os.path.exists(path):
        return []
    out = []
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
        logger.debug("feedback load failed: %s", exc)
    return out


def stats(days: int = 30) -> dict:
    """Compute thumbs ratio + counts for the last ``days`` days.

    Returns ``{"up": int, "down": int, "ratio": float, "total": int}``
    where ratio = up / max(total, 1) in [0, 1].
    """
    cutoff = time.time() - days * 86400
    entries = [e for e in load_all() if e.get("ts", 0) >= cutoff]
    up = sum(1 for e in entries if e.get("rating") == "up")
    down = sum(1 for e in entries if e.get("rating") == "down")
    total = up + down
    return {
        "up": up,
        "down": down,
        "total": total,
        "ratio": (up / total) if total else 0.0,
    }
