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


def test_aggregator_prompt_omits_weights_when_no_reference_names():
    """Backward compat: callers that don't pass reference_models still get
    the same enumerated-response format as before this patch."""
    out = moa._construct_aggregator_prompt(
        "BASE", ["answer one", "answer two"]
    )
    assert "BASE" in out
    assert "1. answer one" in out
    assert "2. answer two" in out
    assert "reliability" not in out  # no annotation


def test_aggregator_prompt_uses_reference_model_names_when_no_metrics(monkeypatch):
    """When reference_models is supplied but no metrics exist yet, the
    prompt still labels each response with its model name (no reliability
    score)."""
    monkeypatch.setattr(
        "agent.moa_metrics.derive_weights",
        lambda names: {n: 0.0 for n in names},
    )
    out = moa._construct_aggregator_prompt(
        "BASE",
        ["answer A", "answer B"],
        reference_models=["modelA", "modelB"],
    )
    # When derive_weights returns 0 for everyone, the wire-up still shows
    # model name + reliability — just both at 0.00
    assert "modelA" in out
    assert "modelB" in out


def test_aggregator_prompt_injects_reliability_scores(monkeypatch):
    """The headline A1 wire-up: derived weights show up in the aggregator
    prompt next to each reference model so the aggregator can weight
    advice by historical reliability."""
    monkeypatch.setattr(
        "agent.moa_metrics.derive_weights",
        lambda names: {"modelA": 0.7, "modelB": 0.2, "modelC": 0.1},
    )
    out = moa._construct_aggregator_prompt(
        "BASE",
        ["good answer", "bad answer", "weird answer"],
        reference_models=["modelA", "modelB", "modelC"],
    )
    assert "modelA" in out and "0.70" in out
    assert "modelB" in out and "0.20" in out
    assert "modelC" in out and "0.10" in out
    # Each response is paired with its model
    assert "good answer" in out
    assert "bad answer" in out
    assert "weird answer" in out


def test_aggregator_prompt_includes_weighting_policy_when_weights_present(monkeypatch):
    """N-A1: when reliability annotations exist, the aggregator also gets
    explicit guidance to prefer majority agreement over a high-reliability
    lone dissent. Without this guidance, A1 alone caused regressions on
    tied prompts (see log-03 in the live G3 re-run)."""
    monkeypatch.setattr(
        "agent.moa_metrics.derive_weights",
        lambda names: {"a": 0.5, "b": 0.3, "c": 0.2},
    )
    out = moa._construct_aggregator_prompt(
        "BASE", ["x", "y", "z"], reference_models=["a", "b", "c"]
    )
    assert "Weighting policy" in out
    assert "majority" in out.lower()
    assert "tie-breaker" in out.lower() or "tie breaker" in out.lower()


def test_aggregator_prompt_omits_policy_without_weights():
    """No weights → no policy section, prompt stays terse."""
    out = moa._construct_aggregator_prompt(
        "BASE", ["a", "b"]
    )
    assert "Weighting policy" not in out


def test_aggregator_prompt_handles_metrics_failure_gracefully(monkeypatch):
    """If agent.moa_metrics raises (e.g. corrupt usage file), the
    aggregator prompt must still build cleanly without reliability
    annotations."""
    def _explode(_names):
        raise RuntimeError("usage file corrupt")
    monkeypatch.setattr("agent.moa_metrics.derive_weights", _explode)
    out = moa._construct_aggregator_prompt(
        "BASE", ["a", "b"], reference_models=["m1", "m2"]
    )
    # Should still contain the responses; just no reliability section
    assert "a" in out and "b" in out
    assert "reliability" not in out


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
