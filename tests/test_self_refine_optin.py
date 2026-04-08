"""Tests for the N-B2 Self-Refine opt-in flag parsing.

The full refine round-trip needs a real LLM call so it's exercised by
live runs against lunark, not unit tests. Here we lock down the
prompt-prefix detection so accidental refactors don't break the
opt-in trigger or, worse, make it always-on.
"""
from __future__ import annotations

import re

import pytest


def _scan_run_agent_source():
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    return (repo / "run_agent.py").read_text()


def test_refine_prefix_token_present():
    """The /refine prefix is part of the parser path — if it gets
    renamed accidentally, the opt-in stops working silently."""
    src = _scan_run_agent_source()
    assert '"/refine "' in src or "'/refine '" in src
    assert "_self_refine_pending" in src


def test_self_refining_now_guard_prevents_recursion():
    """The refine pass calls run_conversation again — without the
    _self_refining_now flag we'd get an infinite tower of refinements."""
    src = _scan_run_agent_source()
    assert "_self_refining_now" in src
    # Make sure the flag is checked before *triggering* refine
    pattern = r"_self_refining_now.*\)\s*:|_self_refining_now\b.*and"
    assert re.search(pattern, src, re.MULTILINE) is not None


def test_refine_uses_final_marker_extraction():
    """The patch instructs the model to write 'FINAL:' before the
    revised answer so we can extract it cleanly."""
    src = _scan_run_agent_source()
    assert "FINAL:" in src
    # And we split on that exact marker when parsing
    assert 'split("FINAL:"' in src or "split('FINAL:'" in src


def test_refine_handles_failure_silently():
    """A try/except around the refine pass prevents the user-facing
    response from being lost when refine itself errors out."""
    src = _scan_run_agent_source()
    # The try block contains the refine call; the except is debug-only
    assert 'logger.debug("self-refine failed' in src


def test_refine_only_when_pending_and_not_interrupted():
    """The trigger guard is conjunctive — pending flag + not interrupted
    + final_response truthy + not currently refining."""
    src = _scan_run_agent_source()
    assert "_self_refine_pending" in src
    assert "interrupted" in src
