"""
Provider Registry — factory functions to get the right provider by config.

Add a new provider? Just add an elif. No plugin systems, no metaclasses.
"""

from __future__ import annotations

import aether.core.config as config_module
from aether.providers.base import LLMProvider, TTSProvider


def get_llm_provider() -> LLMProvider:
    from aether.providers.openai_llm import OpenAILLMProvider

    return OpenAILLMProvider()


def get_tts_provider() -> TTSProvider:
    provider = config_module.config.tts.provider.lower()
    if provider == "openai":
        from aether.providers.openai_tts import OpenAITTSProvider

        return OpenAITTSProvider()
    raise ValueError(f"Unknown TTS provider: {provider}")
