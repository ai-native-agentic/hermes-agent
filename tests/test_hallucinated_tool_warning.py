"""Tests for the silent-failure / hallucinated tool action detector.

The detector lives on AIAgent as _maybe_warn_hallucinated_tool_action and
fires when the model narrates a tool-mediated side effect ("I created the
skill") without actually emitting a tool_call in the current turn. The
warning is best-effort: it's printed for the user but never injected back
into the conversation.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Import lazily so the giant run_agent module only loads once
from run_agent import AIAgent


def _make_agent_stub(session_id: str = "") -> SimpleNamespace:
    """Build the minimum surface AIAgent._maybe_warn_hallucinated_tool_action
    needs to be unit-testable without instantiating a full agent."""
    stub = SimpleNamespace()
    stub.log_prefix = ""
    stub._vprint = MagicMock()
    stub.session_id = session_id
    # Bind the unbound method for direct invocation against the stub.
    stub._maybe_warn_hallucinated_tool_action = (
        AIAgent._maybe_warn_hallucinated_tool_action.__get__(stub)
    )
    stub._HALLUCINATION_VERBS = AIAgent._HALLUCINATION_VERBS
    stub._HALLUCINATION_OBJECTS = AIAgent._HALLUCINATION_OBJECTS
    return stub


def test_warns_when_skill_creation_narrated_without_tool_call():
    agent = _make_agent_stub()
    messages = [
        {"role": "user", "content": "create a haiku skill"},
        {"role": "assistant", "content": "ok"},  # no tool_calls
    ]
    final = "I created the hello-haiku skill in the writing category. Done."
    agent._maybe_warn_hallucinated_tool_action(final, messages)
    assert agent._vprint.called
    msg = agent._vprint.call_args[0][0]
    assert "silent failure" in msg.lower()


def test_no_warning_when_tool_call_actually_happened():
    agent = _make_agent_stub()
    messages = [
        {"role": "user", "content": "create a haiku skill"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "1", "function": {"name": "skill_manage"}}],
        },
        {"role": "tool", "content": '{"success": true}'},
        {"role": "assistant", "content": "I created the hello-haiku skill."},
    ]
    final = "I created the hello-haiku skill."
    agent._maybe_warn_hallucinated_tool_action(final, messages)
    assert not agent._vprint.called


def test_no_warning_for_unrelated_chitchat():
    agent = _make_agent_stub()
    messages = [
        {"role": "user", "content": "what is 17*23?"},
        {"role": "assistant", "content": "391"},
    ]
    final = "391"
    agent._maybe_warn_hallucinated_tool_action(final, messages)
    assert not agent._vprint.called


def test_no_warning_when_only_object_word_present():
    """'skill' alone is not enough — needs an action verb too."""
    agent = _make_agent_stub()
    messages = [
        {"role": "user", "content": "list skills"},
        {"role": "assistant", "content": "ok"},
    ]
    final = "Here are the available skills: foo, bar, baz."
    agent._maybe_warn_hallucinated_tool_action(final, messages)
    assert not agent._vprint.called


def test_no_warning_when_only_verb_present():
    """'created' alone (without a tool object) shouldn't trigger."""
    agent = _make_agent_stub()
    messages = [
        {"role": "user", "content": "tell me about humans"},
        {"role": "assistant", "content": "ok"},
    ]
    final = "Humans created language tens of thousands of years ago."
    agent._maybe_warn_hallucinated_tool_action(final, messages)
    assert not agent._vprint.called


def test_strips_think_blocks_before_matching():
    """Action verbs inside <think>...</think> reasoning are not real claims."""
    agent = _make_agent_stub()
    messages = [
        {"role": "user", "content": "what's 2+2?"},
        {"role": "assistant", "content": "ok"},
    ]
    final = "<think>I should have created a skill for this</think>4"
    agent._maybe_warn_hallucinated_tool_action(final, messages)
    assert not agent._vprint.called


def test_warns_for_memory_save_hallucination():
    agent = _make_agent_stub()
    messages = [
        {"role": "user", "content": "remember that my favorite color is blue"},
        {"role": "assistant", "content": "ok"},
    ]
    final = "I've saved that to memory. I'll remember it for next time."
    agent._maybe_warn_hallucinated_tool_action(final, messages)
    assert agent._vprint.called


def test_warns_for_file_write_hallucination():
    agent = _make_agent_stub()
    messages = [
        {"role": "user", "content": "write a hello world script"},
        {"role": "assistant", "content": "ok"},
    ]
    final = "I wrote the file to /tmp/hello.py for you."
    agent._maybe_warn_hallucinated_tool_action(final, messages)
    assert agent._vprint.called


def test_empty_response_does_not_warn():
    agent = _make_agent_stub()
    agent._maybe_warn_hallucinated_tool_action("", [])
    assert not agent._vprint.called


def test_warning_includes_resume_hint_when_session_id_set():
    """The user should get a copy-pasteable command to retry the turn with
    an explicit "use the tool" instruction. Without the session id we
    can't build it, so it's only included when one is available."""
    agent = _make_agent_stub(session_id="20260408_141529_bd77df")
    messages = [
        {"role": "user", "content": "save my favorite color"},
        {"role": "assistant", "content": "ok"},
    ]
    agent._maybe_warn_hallucinated_tool_action(
        "I saved that to memory.", messages,
    )
    assert agent._vprint.called
    msg = agent._vprint.call_args[0][0]
    assert "hermes --resume 20260408_141529_bd77df" in msg
    assert "do not narrate" in msg.lower() or "actually call" in msg.lower()


def test_warning_omits_resume_hint_without_session_id():
    """No session id → just print the warning, no retry hint."""
    agent = _make_agent_stub(session_id="")
    messages = [
        {"role": "user", "content": "save my color"},
        {"role": "assistant", "content": "ok"},
    ]
    agent._maybe_warn_hallucinated_tool_action(
        "I saved that to memory.", messages,
    )
    assert agent._vprint.called
    msg = agent._vprint.call_args[0][0]
    assert "hermes --resume" not in msg
