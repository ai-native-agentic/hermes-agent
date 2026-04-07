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
