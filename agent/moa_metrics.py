"""Per-model invocation metrics for the Mixture-of-Agents tool.

Mirrors agent/skill_metrics.py but tracks LLMs instead of skills. The
goal is to feed `tools/mixture_of_agents_tool.py`'s reference / aggregator
selection (and the example runners' weighted-vote logic) with empirical
success rates instead of hand-tuned constants.

Storage: ``$HERMES_HOME/moa_metrics.json``::

    {
      "Qwen3-32B": {
        "calls": 42,            # total invocations
        "successes": 41,        # responses that returned non-empty content
        "last_call": 1775568002
      },
      ...
    }

The default weight derivation is intentionally simple and explainable::

    weight(model) = log(1 + calls) * (successes / max(calls, 1))

This favors (a) models we have *enough data* on, and (b) models that
actually succeed. New models with no data fall back to a uniform weight
so they're not punished for being new. All weights are L1-normalized so
the caller can use them as voting multipliers directly.

Failure modes are silent: any IO error logs at DEBUG and the call
returns gracefully. Metrics must never block the actual MoA pipeline.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def _metrics_path() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home() / "moa_metrics.json")


def _load() -> Dict[str, Dict[str, float]]:
    path = _metrics_path()
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}


def _save_atomic(data: Dict[str, Dict[str, float]]) -> None:
    path = _metrics_path()
    parent = os.path.dirname(path)
    try:
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".moa_metrics.", suffix=".tmp", dir=parent)
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
        logger.debug("moa metrics save failed: %s", exc)


def record_call(model: str, success: bool) -> None:
    """Best-effort: bump call/success counters for ``model``."""
    if not model:
        return
    now = int(time.time())
    with _LOCK:
        data = _load()
        entry = data.setdefault(model, {})
        entry["calls"] = int(entry.get("calls", 0)) + 1
        if success:
            entry["successes"] = int(entry.get("successes", 0)) + 1
        entry["last_call"] = now
        _save_atomic(data)


def load_metrics() -> Dict[str, Dict[str, float]]:
    with _LOCK:
        return dict(_load())


def derive_weights(
    model_names: Iterable[str],
    *,
    default_weight: float = 1.0,
    floor: float = 0.1,
) -> Dict[str, float]:
    """Compute L1-normalized voting weights for the given model names.

    Models with no recorded calls get ``default_weight`` so they aren't
    locked out by lack of data. Each weight is then floored at ``floor``
    relative to the max so a single bad day can't drop a model to zero.
    Returns ``{model: weight}`` summing to 1.0 (or empty dict if no input).
    """
    names = list(model_names)
    if not names:
        return {}
    data = load_metrics()
    raw: Dict[str, float] = {}
    for name in names:
        entry = data.get(name) or {}
        calls = float(entry.get("calls", 0))
        successes = float(entry.get("successes", 0))
        if calls <= 0:
            raw[name] = default_weight
            continue
        success_rate = successes / calls if calls > 0 else 0.0
        # log smoothing favors models with more data without exploding
        raw[name] = math.log(1 + calls) * success_rate
    # Apply floor relative to max
    max_w = max(raw.values()) if raw else 0.0
    if max_w > 0:
        floor_value = max_w * floor
        raw = {k: max(v, floor_value) for k, v in raw.items()}
    # L1 normalize
    total = sum(raw.values())
    if total == 0:
        return {k: 1.0 / len(names) for k in names}
    return {k: v / total for k, v in raw.items()}
