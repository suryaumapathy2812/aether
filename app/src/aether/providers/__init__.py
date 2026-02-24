"""
Aether Providers — abstract interfaces for LLM and TTS.

Each provider interface defines the contract. Concrete implementations
(OpenAI, etc.) live alongside. Swap providers by changing config.
"""

from aether.providers.base import LLMProvider, TTSProvider
from aether.providers.registry import get_llm_provider, get_tts_provider

__all__ = [
    "LLMProvider",
    "TTSProvider",
    "get_llm_provider",
    "get_tts_provider",
]
