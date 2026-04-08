"""Capability gating: drop tools when /v1/models advertises tool_calling=false.

This is the runtime counterpart to the metadata extraction added in P1
(commit 74e8622f). The agent must consume the flag, not just expose it.
"""
from __future__ import annotations
from unittest.mock import patch

import pytest


def _build_agent(model: str, base_url: str, fake_meta: dict):
    """Construct an AIAgent with the metadata cache mocked.

    Returns the constructed AIAgent so the test can inspect agent.tools.
    """
    from run_agent import AIAgent

    def _fake_fetch(base, api_key="", force_refresh=False):
        return fake_meta

    with patch("agent.model_metadata.fetch_endpoint_model_metadata", side_effect=_fake_fetch):
        agent = AIAgent(
            base_url=base_url,
            api_key="x",
            provider="lunark",
            api_mode="chat_completions",
            model=model,
            quiet_mode=True,
            persist_session=False,
            skip_memory=True,
            skip_context_files=True,
            enabled_toolsets=["hermes-cli"],
        )
    return agent


def test_tools_dropped_when_model_advertises_no_tool_calling():
    """Lunark vLLM models like Qwen3.5-27B advertise tool_calling=false. The
    agent should not ship tool definitions to them — doing so just produces
    silent-failure narration ("I created the skill" with no tool_call)."""
    fake_meta = {
        "Qwen3.5-27B": {
            "name": "Qwen3.5-27B",
            "context_length": 65536,
            "tool_calling": False,
        }
    }
    agent = _build_agent("Qwen3.5-27B", "https://llm.lunark.ai/v1", fake_meta)
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_tools_kept_when_model_advertises_tool_calling_true():
    fake_meta = {
        "Qwen3-32B": {
            "name": "Qwen3-32B",
            "context_length": 32768,
            "tool_calling": True,
        }
    }
    agent = _build_agent("Qwen3-32B", "https://llm.lunark.ai/v1", fake_meta)
    assert len(agent.tools) > 0
    assert agent.valid_tool_names  # non-empty


def test_tools_kept_when_metadata_missing_the_flag():
    """Backward compat: most providers don't expose tool_calling at all.
    In that case the gate must NOT remove tools (would break OpenRouter)."""
    fake_meta = {
        "Qwen3-32B": {
            "name": "Qwen3-32B",
            "context_length": 32768,
            # no tool_calling field
        }
    }
    agent = _build_agent("Qwen3-32B", "https://llm.lunark.ai/v1", fake_meta)
    assert len(agent.tools) > 0


def test_tools_kept_when_model_not_in_metadata():
    """If the probe returns metadata but our model isn't listed, behave the
    same as the missing-flag case — keep tools."""
    fake_meta = {"some-other-model": {"tool_calling": False}}
    agent = _build_agent("Qwen3-32B", "https://llm.lunark.ai/v1", fake_meta)
    assert len(agent.tools) > 0
