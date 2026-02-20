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
    # Separate model for voice — gpt-4o-mini is ~1s faster TTFT, good for voice
    voice_model: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.7
    max_history_turns: int = 20
    # Base URL for OpenAI-compatible APIs (e.g., OpenRouter)
    # OpenRouter: https://openrouter.ai/api/v1
    base_url: str = ""

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            provider=os.getenv("AETHER_LLM_PROVIDER", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("AETHER_LLM_MODEL", "gpt-4o"),
            voice_model=os.getenv("AETHER_LLM_VOICE_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("AETHER_LLM_MAX_TOKENS", "500")),
            temperature=float(os.getenv("AETHER_LLM_TEMPERATURE", "0.7")),
            max_history_turns=int(os.getenv("AETHER_LLM_MAX_HISTORY_TURNS", "20")),
            base_url=os.getenv("OPENAI_BASE_URL", ""),
        )


@dataclass(frozen=True)
class MemoryConfig:
    """Memory store settings."""

    db_path: str = "aether_memory.db"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    similarity_threshold: float = 0.3
    search_limit: int = 5
    # v0.07: action memory + session summaries
    action_retention_days: int = 7
    session_summary_limit: int = 3
    action_output_max_chars: int = 1000

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
            action_retention_days=int(os.getenv("AETHER_ACTION_RETENTION_DAYS", "7")),
            session_summary_limit=int(os.getenv("AETHER_SESSION_SUMMARY_LIMIT", "3")),
            action_output_max_chars=int(
                os.getenv("AETHER_ACTION_OUTPUT_MAX_CHARS", "1000")
            ),
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
class PersonalityConfig:
    """User personality / style settings."""

    base_style: str = "default"  # default, concise, detailed, friendly, professional
    custom_instructions: str = ""

    @classmethod
    def from_env(cls) -> PersonalityConfig:
        return cls(
            base_style=os.getenv("AETHER_BASE_STYLE", "default"),
            custom_instructions=os.getenv("AETHER_CUSTOM_INSTRUCTIONS", ""),
        )


@dataclass(frozen=True)
class ServerConfig:
    """Server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    ws_send_timeout: float = 5.0
    debounce_delay: float = 0.3
    working_dir: str = "/"  # Docker default: full access. Container is the sandbox.

    @classmethod
    def from_env(cls) -> ServerConfig:
        return cls(
            host=os.getenv("AETHER_HOST", "0.0.0.0"),
            port=int(os.getenv("AETHER_PORT", "8000")),
            ws_send_timeout=float(os.getenv("AETHER_WS_SEND_TIMEOUT", "5.0")),
            debounce_delay=float(os.getenv("AETHER_DEBOUNCE_DELAY", "0.3")),
            working_dir=os.getenv("AETHER_WORKING_DIR", "/"),
        )


@dataclass(frozen=True)
class KernelConfig:
    """Kernel scheduler settings."""

    workers_interactive: int = 2  # P-Core workers (user-facing replies)
    workers_background: int = 2  # E-Core workers (memory, notifications)
    interactive_queue_limit: int = 20  # Raise QueueFullError when exceeded
    background_queue_limit: int = 50  # Shed oldest job when exceeded

    @classmethod
    def from_env(cls) -> "KernelConfig":
        return cls(
            workers_interactive=int(
                os.getenv("AETHER_KERNEL_WORKERS_INTERACTIVE", "2")
            ),
            workers_background=int(os.getenv("AETHER_KERNEL_WORKERS_BACKGROUND", "2")),
            interactive_queue_limit=int(
                os.getenv("AETHER_INTERACTIVE_QUEUE_LIMIT", "20")
            ),
            background_queue_limit=int(
                os.getenv("AETHER_BACKGROUND_QUEUE_LIMIT", "50")
            ),
        )


@dataclass(frozen=True)
class VADConfig:
    """Voice activity detection settings (Silero ONNX)."""

    mode: str = "off"  # off | shadow | active
    model_path: str = ""
    sample_rate: int = 16000
    activation_threshold: float = 0.5
    deactivation_threshold: float = 0.35
    min_speech_duration: float = 0.05
    min_silence_duration: float = 0.55
    # STT gating — connect STT only when speech detected
    stt_pre_roll_ms: int = 500  # ring buffer size (ms of audio to keep)
    stt_idle_disconnect_s: float = 5.0  # seconds of silence before disconnecting STT

    @classmethod
    def from_env(cls) -> "VADConfig":
        return cls(
            mode=os.getenv("AETHER_VAD_MODE", "off").lower(),
            model_path=os.getenv("AETHER_VAD_MODEL_PATH", ""),
            sample_rate=int(os.getenv("AETHER_VAD_SAMPLE_RATE", "16000")),
            activation_threshold=float(
                os.getenv("AETHER_VAD_ACTIVATION_THRESHOLD", "0.5")
            ),
            deactivation_threshold=float(
                os.getenv("AETHER_VAD_DEACTIVATION_THRESHOLD", "0.35")
            ),
            min_speech_duration=float(
                os.getenv("AETHER_VAD_MIN_SPEECH_DURATION", "0.05")
            ),
            min_silence_duration=float(
                os.getenv("AETHER_VAD_MIN_SILENCE_DURATION", "0.55")
            ),
            stt_pre_roll_ms=int(os.getenv("AETHER_STT_PRE_ROLL_MS", "500")),
            stt_idle_disconnect_s=float(
                os.getenv("AETHER_STT_IDLE_DISCONNECT_S", "5.0")
            ),
        )


@dataclass(frozen=True)
class WebRTCConfig:
    """WebRTC transport settings."""

    session_ttl_seconds: int = 600  # 10 min idle before session cleanup
    disconnect_grace_seconds: int = 10  # grace period before pausing on disconnect

    @classmethod
    def from_env(cls) -> "WebRTCConfig":
        return cls(
            session_ttl_seconds=int(os.getenv("AETHER_WEBRTC_SESSION_TTL", "600")),
            disconnect_grace_seconds=int(
                os.getenv("AETHER_WEBRTC_DISCONNECT_GRACE", "10")
            ),
        )


@dataclass(frozen=True)
class TelephonyConfig:
    """Telephony transport settings (base settings, provider-specific config via plugins)."""

    enabled: bool = False
    sample_rate: int = 8000
    encoding: str = "mulaw"

    @classmethod
    def from_env(cls) -> "TelephonyConfig":
        return cls(
            enabled=os.getenv("AETHER_TELEPHONY_ENABLED", "false").lower() == "true",
            sample_rate=int(os.getenv("AETHER_TELEPHONY_SAMPLE_RATE", "8000")),
            encoding=os.getenv("AETHER_TELEPHONY_ENCODING", "mulaw"),
        )


@dataclass(frozen=True)
class TurnDetectionConfig:
    """LiveKit turn detector settings."""

    mode: str = "off"  # off | shadow | active
    model_type: str = "en"  # en | multilingual
    model_repo: str = "livekit/turn-detector"
    # v1.2.2-en has model.onnx (float32) — correct on ARM/aarch64.
    # v0.4.1-intl only has model_q8.onnx (INT8/x86) — broken on ARM.
    model_revision: str = "v1.2.2-en"
    model_filename: str = "model.onnx"
    model_dir: str = "/models/turn-detector"
    min_endpointing_delay: float = 0.3
    max_endpointing_delay: float = 3.0
    inference_timeout_seconds: float = 0.35

    @classmethod
    def from_env(cls) -> "TurnDetectionConfig":
        model_type = os.getenv("AETHER_TURN_MODEL_TYPE", "en").lower()
        default_revision = "v1.2.2-en" if model_type == "en" else "v0.4.1-intl"
        default_filename = "model.onnx" if model_type == "en" else "model_q8.onnx"
        return cls(
            mode=os.getenv("AETHER_TURN_DETECTION_MODE", "off").lower(),
            model_type=model_type,
            model_repo=os.getenv("AETHER_TURN_MODEL_REPO", "livekit/turn-detector"),
            model_revision=os.getenv("AETHER_TURN_MODEL_REVISION", default_revision),
            model_filename=os.getenv("AETHER_TURN_MODEL_FILENAME", default_filename),
            model_dir=os.getenv("AETHER_TURN_MODEL_DIR", "/models/turn-detector"),
            min_endpointing_delay=float(
                os.getenv("AETHER_TURN_MIN_ENDPOINTING_DELAY", "0.3")
            ),
            max_endpointing_delay=float(
                os.getenv("AETHER_TURN_MAX_ENDPOINTING_DELAY", "3.0")
            ),
            inference_timeout_seconds=float(
                os.getenv("AETHER_TURN_INFERENCE_TIMEOUT_SECONDS", "2.0")
            ),
        )


@dataclass(frozen=True)
class AetherConfig:
    """Root configuration — one object to rule them all."""

    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    webrtc: WebRTCConfig = field(default_factory=WebRTCConfig)
    telephony: TelephonyConfig = field(default_factory=TelephonyConfig)
    turn_detection: TurnDetectionConfig = field(default_factory=TurnDetectionConfig)

    @classmethod
    def from_env(cls) -> AetherConfig:
        return cls(
            stt=STTConfig.from_env(),
            llm=LLMConfig.from_env(),
            tts=TTSConfig.from_env(),
            memory=MemoryConfig.from_env(),
            personality=PersonalityConfig.from_env(),
            server=ServerConfig.from_env(),
            kernel=KernelConfig.from_env(),
            vad=VADConfig.from_env(),
            webrtc=WebRTCConfig.from_env(),
            telephony=TelephonyConfig.from_env(),
            turn_detection=TurnDetectionConfig.from_env(),
        )


# Singleton — import this wherever you need config
config = AetherConfig.from_env()


def reload_config() -> AetherConfig:
    """Reload config from environment. Returns the new config.

    Used when the orchestrator signals a preference change.
    Since dataclasses are frozen, we replace the global singleton.
    """
    global config
    config = AetherConfig.from_env()
    return config
