import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

moa = importlib.import_module("tools.mixture_of_agents_tool")


def test_moa_defaults_track_current_openrouter_frontier_models():
    assert moa.REFERENCE_MODELS == [
        "anthropic/claude-opus-4.6",
        "google/gemini-3-pro-preview",
        "openai/gpt-5.4-pro",
        "deepseek/deepseek-v3.2",
    ]
    assert moa.AGGREGATOR_MODEL == "anthropic/claude-opus-4.6"


@pytest.mark.asyncio
async def test_reference_model_retry_warnings_avoid_exc_info_until_terminal_failure(monkeypatch):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=RuntimeError("rate limited"))
            )
        )
    )
    warn = MagicMock()
    err = MagicMock()

    monkeypatch.setattr(moa, "_get_openrouter_client", lambda: fake_client)
    monkeypatch.setattr(moa.logger, "warning", warn)
    monkeypatch.setattr(moa.logger, "error", err)

    model, message, success = await moa._run_reference_model_safe(
        "openai/gpt-5.4-pro", "hello", max_retries=2
    )

    assert model == "openai/gpt-5.4-pro"
    assert success is False
    assert "failed after 2 attempts" in message
    assert warn.call_count == 2
    assert all(call.kwargs.get("exc_info") is None for call in warn.call_args_list)
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True


def test_resolve_models_falls_back_to_openrouter_defaults(monkeypatch):
    """When config.yaml has no `moa:` section, the resolver returns the
    hard-coded OpenRouter REFERENCE_MODELS / AGGREGATOR_MODEL constants
    so backward compatibility is preserved."""
    monkeypatch.setattr(moa, "_load_moa_config", lambda: {})
    refs, agg, ref_t, agg_t, enable_reasoning = moa._resolve_models()
    assert refs == moa.REFERENCE_MODELS
    assert agg == moa.AGGREGATOR_MODEL
    assert ref_t == moa.REFERENCE_TEMPERATURE
    assert agg_t == moa.AGGREGATOR_TEMPERATURE
    # OpenRouter is the default → reasoning extra_body enabled
    assert enable_reasoning is True


def test_resolve_models_uses_lunark_config(monkeypatch):
    """Custom provider config: should swap reference list, aggregator, and
    disable the OpenRouter-specific reasoning extra_body by default."""
    monkeypatch.setattr(moa, "_load_moa_config", lambda: {
        "provider": "custom",
        "base_url": "https://llm.lunark.ai/v1",
        "api_key_env": "LUNARK_API_KEY",
        "reference_models": ["Qwen3.5-27B", "Qwen2.5-32B-Instruct", "Gemma-4-E4B-it"],
        "aggregator_model": "Qwen3-32B",
    })
    refs, agg, _, _, enable_reasoning = moa._resolve_models()
    assert refs == ["Qwen3.5-27B", "Qwen2.5-32B-Instruct", "Gemma-4-E4B-it"]
    assert agg == "Qwen3-32B"
    # Custom provider → no extra_body reasoning by default
    assert enable_reasoning is False


def test_check_moa_requirements_uses_configured_api_key_env(monkeypatch):
    """For a custom provider, check_moa_requirements should look at the
    env var named in moa.api_key_env, not OPENROUTER_API_KEY."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LUNARK_API_KEY", raising=False)

    monkeypatch.setattr(moa, "_load_moa_config", lambda: {
        "provider": "custom",
        "api_key_env": "LUNARK_API_KEY",
    })

    # No key set → False
    assert moa.check_moa_requirements() is False

    monkeypatch.setenv("LUNARK_API_KEY", "vllm-local")
    assert moa.check_moa_requirements() is True


def test_get_moa_configuration_reflects_active_provider(monkeypatch):
    """get_moa_configuration() should expose the active provider, models,
    and capability flags so callers can introspect the runtime config."""
    monkeypatch.setattr(moa, "_load_moa_config", lambda: {
        "provider": "custom",
        "base_url": "https://llm.lunark.ai/v1",
        "reference_models": ["Qwen3.5-27B"],
        "aggregator_model": "Qwen3-32B",
        "reference_temperature": 0.5,
        "aggregator_temperature": 0.3,
    })
    info = moa.get_moa_configuration()
    assert info["provider"] == "custom"
    assert info["reference_models"] == ["Qwen3.5-27B"]
    assert info["aggregator_model"] == "Qwen3-32B"
    assert info["reference_temperature"] == 0.5
    assert info["aggregator_temperature"] == 0.3
    assert info["enable_reasoning_extra_body"] is False


@pytest.mark.asyncio
async def test_moa_top_level_error_logs_single_traceback_on_aggregator_failure(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        moa,
        "_run_reference_model_safe",
        AsyncMock(return_value=("anthropic/claude-opus-4.6", "ok", True)),
    )
    monkeypatch.setattr(
        moa,
        "_run_aggregator_model",
        AsyncMock(side_effect=RuntimeError("aggregator boom")),
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    err = MagicMock()
    monkeypatch.setattr(moa.logger, "error", err)

    result = json.loads(
        await moa.mixture_of_agents_tool(
            "solve this",
            reference_models=["anthropic/claude-opus-4.6"],
        )
    )

    assert result["success"] is False
    assert "Error in MoA processing" in result["error"]
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True
