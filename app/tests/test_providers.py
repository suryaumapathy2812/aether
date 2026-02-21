"""Tests for provider interfaces and registry."""

import pytest
import aether.core.config as config_module
from aether.core.config import reload_config
from aether.providers.base import STTProvider, LLMProvider, TTSProvider
from aether.providers.registry import (
    get_stt_provider,
    get_llm_provider,
    get_tts_provider,
)
from aether.providers.openai_llm import _get_model_name


def test_stt_provider_is_abstract():
    """Can't instantiate STTProvider directly."""
    with pytest.raises(TypeError):
        STTProvider()  # type: ignore


def test_llm_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore


def test_tts_provider_is_abstract():
    with pytest.raises(TypeError):
        TTSProvider()  # type: ignore


def test_get_stt_provider_returns_deepgram():
    provider = get_stt_provider()
    assert provider.__class__.__name__ == "DeepgramSTTProvider"


def test_get_llm_provider_returns_openai():
    provider = get_llm_provider()
    assert provider.__class__.__name__ == "OpenAILLMProvider"


def test_get_tts_provider_returns_openai():
    provider = get_tts_provider()
    assert provider.__class__.__name__ == "OpenAITTSProvider"


def test_openai_model_name_uses_reloaded_config(monkeypatch):
    """Model names are always prefixed for OpenRouter (hardcoded base_url)."""
    previous = config_module.config
    monkeypatch.setenv("AETHER_LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    reload_config()
    assert _get_model_name() == "openai/gpt-4.1-mini"

    # Restore original config object after the test.
    config_module.config = previous
