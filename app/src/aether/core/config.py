"""
Aether Configuration — single source of truth for all settings.

Reads from environment variables with sensible defaults.
No config files, no YAML, no complexity. Just env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class STTConfig:
    """Speech-to-text provider settings."""

    provider: str = "deepgram"
    api_key: str = ""
    model: str = "nova-3"
    language: str = "en"
    sample_rate: int = 16000
    encoding: str = "linear16"
    utterance_end_ms: int = 1200
    endpointing_ms: int = 300
    # Reconnection
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0  # seconds, doubles each attempt

    @classmethod
    def from_env(cls) -> STTConfig:
        return cls(
            provider=os.getenv("AETHER_STT_PROVIDER", "deepgram"),
            api_key=os.getenv("DEEPGRAM_API_KEY", ""),
            model=os.getenv("AETHER_STT_MODEL", "nova-3"),
            language=os.getenv("AETHER_STT_LANGUAGE", "en"),
            sample_rate=int(os.getenv("AETHER_STT_SAMPLE_RATE", "16000")),
            encoding=os.getenv("AETHER_STT_ENCODING", "linear16"),
            utterance_end_ms=int(os.getenv("AETHER_STT_UTTERANCE_END_MS", "1200")),
            endpointing_ms=int(os.getenv("AETHER_STT_ENDPOINTING_MS", "300")),
            reconnect_attempts=int(os.getenv("AETHER_STT_RECONNECT_ATTEMPTS", "5")),
            reconnect_delay=float(os.getenv("AETHER_STT_RECONNECT_DELAY", "1.0")),
        )


@dataclass(frozen=True)
class LLMConfig:
    """Language model provider settings."""

    provider: str = "openai"
    api_key: str = ""
    model: str = "gpt-4o"
    max_tokens: int = 500
    temperature: float = 0.7
    max_history_turns: int = 20

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            provider=os.getenv("AETHER_LLM_PROVIDER", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("AETHER_LLM_MODEL", "gpt-4o"),
            max_tokens=int(os.getenv("AETHER_LLM_MAX_TOKENS", "500")),
            temperature=float(os.getenv("AETHER_LLM_TEMPERATURE", "0.7")),
            max_history_turns=int(os.getenv("AETHER_LLM_MAX_HISTORY_TURNS", "20")),
        )


@dataclass(frozen=True)
class MemoryConfig:
    """Memory store settings."""

    db_path: str = "aether_memory.db"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    similarity_threshold: float = 0.3
    search_limit: int = 5

    @classmethod
    def from_env(cls) -> MemoryConfig:
        return cls(
            db_path=os.getenv("AETHER_DB_PATH", "aether_memory.db"),
            embedding_model=os.getenv(
                "AETHER_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            embedding_dim=int(os.getenv("AETHER_EMBEDDING_DIM", "1536")),
            similarity_threshold=float(os.getenv("AETHER_MEMORY_THRESHOLD", "0.3")),
            search_limit=int(os.getenv("AETHER_MEMORY_SEARCH_LIMIT", "5")),
        )


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech provider settings."""

    provider: str = "openai"
    api_key: str = ""
    model: str = "tts-1"
    voice: str = "nova"
    timeout: float = 15.0
    # Sarvam (Bulbul v3)
    sarvam_api_key: str = ""
    sarvam_model: str = "bulbul:v3"
    sarvam_speaker: str = "shubh"
    sarvam_language: str = "en-IN"
    sarvam_sample_rate: int = 24000
    # ElevenLabs
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel

    @classmethod
    def from_env(cls) -> TTSConfig:
        return cls(
            provider=os.getenv("AETHER_TTS_PROVIDER", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("AETHER_TTS_MODEL", "tts-1"),
            voice=os.getenv("AETHER_TTS_VOICE", "nova"),
            timeout=float(os.getenv("AETHER_TTS_TIMEOUT", "15.0")),
            # Sarvam
            sarvam_api_key=os.getenv("SARVAM_API_KEY", ""),
            sarvam_model=os.getenv("AETHER_SARVAM_MODEL", "bulbul:v3"),
            sarvam_speaker=os.getenv("AETHER_SARVAM_SPEAKER", "shubh"),
            sarvam_language=os.getenv("AETHER_SARVAM_LANGUAGE", "en-IN"),
            sarvam_sample_rate=int(os.getenv("AETHER_SARVAM_SAMPLE_RATE", "24000")),
            # ElevenLabs
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            elevenlabs_model=os.getenv(
                "AETHER_ELEVENLABS_MODEL", "eleven_multilingual_v2"
            ),
            elevenlabs_voice_id=os.getenv(
                "AETHER_ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"
            ),
        )


@dataclass(frozen=True)
class ServerConfig:
    """Server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    ws_send_timeout: float = 5.0
    debounce_delay: float = 1.5
    working_dir: str = "/"  # Docker default: full access. Container is the sandbox.

    @classmethod
    def from_env(cls) -> ServerConfig:
        return cls(
            host=os.getenv("AETHER_HOST", "0.0.0.0"),
            port=int(os.getenv("AETHER_PORT", "8000")),
            ws_send_timeout=float(os.getenv("AETHER_WS_SEND_TIMEOUT", "5.0")),
            debounce_delay=float(os.getenv("AETHER_DEBOUNCE_DELAY", "1.5")),
            working_dir=os.getenv("AETHER_WORKING_DIR", "/"),
        )


@dataclass(frozen=True)
class AetherConfig:
    """Root configuration — one object to rule them all."""

    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_env(cls) -> AetherConfig:
        return cls(
            stt=STTConfig.from_env(),
            llm=LLMConfig.from_env(),
            tts=TTSConfig.from_env(),
            memory=MemoryConfig.from_env(),
            server=ServerConfig.from_env(),
        )


# Singleton — import this wherever you need config
config = AetherConfig.from_env()
