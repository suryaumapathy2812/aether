"""
Aether Providers — abstract interfaces for STT, LLM, and TTS.

Each provider interface defines the contract. Concrete implementations
(Deepgram, OpenAI, etc.) live alongside. Swap providers by changing config.
"""

from aether.providers.base import STTProvider, LLMProvider, TTSProvider
from aether.providers.registry import get_stt_provider, get_llm_provider, get_tts_provider

__all__ = [
    "STTProvider",
    "LLMProvider",
    "TTSProvider",
    "get_stt_provider",
    "get_llm_provider",
    "get_tts_provider",
]
