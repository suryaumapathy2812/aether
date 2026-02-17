"""Tests for provider interfaces and registry."""

import pytest
from aether.providers.base import STTProvider, LLMProvider, TTSProvider
from aether.providers.registry import get_stt_provider, get_llm_provider, get_tts_provider


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
