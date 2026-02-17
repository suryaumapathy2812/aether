"""
Provider Registry — factory functions to get the right provider by config.

Add a new provider? Just add an elif. No plugin systems, no metaclasses.
"""

from __future__ import annotations

from aether.core.config import config
from aether.providers.base import STTProvider, LLMProvider, TTSProvider


def get_stt_provider() -> STTProvider:
    provider = config.stt.provider.lower()
    if provider == "deepgram":
        from aether.providers.deepgram_stt import DeepgramSTTProvider
        return DeepgramSTTProvider()
    raise ValueError(f"Unknown STT provider: {provider}")


def get_llm_provider() -> LLMProvider:
    provider = config.llm.provider.lower()
    if provider == "openai":
        from aether.providers.openai_llm import OpenAILLMProvider
        return OpenAILLMProvider()
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_tts_provider() -> TTSProvider:
    provider = config.tts.provider.lower()
    if provider == "openai":
        from aether.providers.openai_tts import OpenAITTSProvider
        return OpenAITTSProvider()
    raise ValueError(f"Unknown TTS provider: {provider}")
