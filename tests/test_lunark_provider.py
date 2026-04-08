"""Tests for the Lunark (vLLM) provider overlay."""
from hermes_cli.providers import (
    HERMES_OVERLAYS,
    determine_api_mode,
    get_label,
    get_provider,
    normalize_provider,
)


def test_lunark_overlay_registered():
    assert "lunark" in HERMES_OVERLAYS
    overlay = HERMES_OVERLAYS["lunark"]
    assert overlay.transport == "openai_chat"
    assert overlay.base_url_override == "https://llm.lunark.ai/v1"
    assert "LUNARK_API_KEY" in overlay.extra_env_vars


def test_lunark_aliases_normalize():
    assert normalize_provider("lunark") == "lunark"
    assert normalize_provider("Lunark.AI") == "lunark"
    assert normalize_provider("lunark-ai") == "lunark"


def test_lunark_get_provider_returns_def():
    pdef = get_provider("lunark")
    assert pdef is not None
    assert pdef.id == "lunark"
    assert pdef.transport == "openai_chat"
    assert pdef.base_url == "https://llm.lunark.ai/v1"
    assert pdef.base_url_env_var == "LUNARK_BASE_URL"
    assert "LUNARK_API_KEY" in pdef.api_key_env_vars
    assert pdef.is_aggregator is True


def test_lunark_label():
    assert get_label("lunark") == "Lunark (vLLM)"
    assert get_label("lunark.ai") == "Lunark (vLLM)"


def test_lunark_api_mode_is_chat_completions():
    assert determine_api_mode("lunark") == "chat_completions"


def test_endpoint_metadata_extracts_capability_flags(monkeypatch):
    """Lunark exposes tool_calling/reasoning booleans on each model.
    Hermes' fetch_endpoint_model_metadata should surface them so other
    parts of the agent (e.g. tool injection, reasoning channel handling)
    can branch on real capabilities instead of guessing."""
    from agent import model_metadata

    fake_payload = {
        "object": "list",
        "data": [
            {
                "id": "Qwen3-32B",
                "object": "model",
                "owned_by": "vllm",
                "context_length": 32768,
                "max_model_len": 32768,
                "tool_calling": True,
                "reasoning": False,
            },
            {
                "id": "Gemma-4-E4B-it",
                "object": "model",
                "owned_by": "vllm",
                "context_length": 32768,
                "max_model_len": 32768,
                "tool_calling": True,
                "reasoning": True,
            },
        ],
    }

    class _FakeResp:
        ok = True
        status_code = 200

        def json(self):
            return fake_payload

        def raise_for_status(self):
            pass

    def _fake_get(url, headers=None, timeout=None):
        return _FakeResp()

    monkeypatch.setattr(model_metadata.requests, "get", _fake_get)
    # Bust both caches
    model_metadata._endpoint_model_metadata_cache.clear()
    model_metadata._endpoint_model_metadata_cache_time.clear()

    meta = model_metadata.fetch_endpoint_model_metadata(
        "https://llm.lunark.ai/v1", api_key="x", force_refresh=True
    )

    assert "Qwen3-32B" in meta
    assert meta["Qwen3-32B"]["context_length"] == 32768
    assert meta["Qwen3-32B"]["tool_calling"] is True
    assert meta["Qwen3-32B"]["reasoning"] is False

    assert meta["Gemma-4-E4B-it"]["reasoning"] is True
    assert meta["Gemma-4-E4B-it"]["tool_calling"] is True
