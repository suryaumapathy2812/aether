"""
Provider Registry — factory functions to get the right provider by config.

Add a new provider? Just add an elif. No plugin systems, no metaclasses.
"""

from __future__ import annotations

import aether.core.config as config_module
from aether.providers.base import STTProvider, LLMProvider, TTSProvider


def get_stt_provider() -> STTProvider:
    provider = config_module.config.stt.provider.lower()
    if provider == "deepgram":
        from aether.providers.deepgram_stt import DeepgramSTTProvider

        return DeepgramSTTProvider()
    raise ValueError(f"Unknown STT provider: {provider}")


def get_llm_provider() -> LLMProvider:
    provider = config_module.config.llm.provider.lower()
    if provider == "openai":
        from aether.providers.openai_llm import OpenAILLMProvider

        return OpenAILLMProvider()
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_tts_provider() -> TTSProvider:
    provider = config_module.config.tts.provider.lower()
    if provider == "openai":
        from aether.providers.openai_tts import OpenAITTSProvider

        return OpenAITTSProvider()
    elif provider == "sarvam":
        from aether.providers.sarvam_tts import SarvamTTSProvider

        return SarvamTTSProvider()
    elif provider == "elevenlabs":
        from aether.providers.elevenlabs_tts import ElevenLabsTTSProvider

        return ElevenLabsTTSProvider()
    raise ValueError(f"Unknown TTS provider: {provider}")
