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
class LLMConfig:
    """Language model provider settings.

    LLM traffic always routes through OpenRouter (hardcoded base URL).
    OPENROUTER_API_KEY is the only key needed for LLM calls.
    OPENAI_API_KEY is used separately for TTS and embeddings (direct OpenAI).
    """

    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    api_key: str = ""
    model: str = "openai/gpt-4o"
    # Separate model for voice — gpt-4o-mini is ~1s faster TTFT, good for voice
    voice_model: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.7
    max_history_turns: int = 20
    # Always OpenRouter — hardcoded, not configurable via env.
    base_url: str = OPENROUTER_BASE_URL
    # Derived from model name — used for logging/display only.
    provider: str = "openai"

    @classmethod
    def from_env(cls) -> LLMConfig:
        model = os.getenv("AETHER_LLM_MODEL", "openai/gpt-4o").strip()
        api_key = os.getenv("OPENROUTER_API_KEY", "")

        # Derive provider from model name for logging/display.
        provider = _infer_provider_from_model(model)

        return cls(
            api_key=api_key,
            model=model,
            provider=provider,
            voice_model=os.getenv("AETHER_LLM_VOICE_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("AETHER_LLM_MAX_TOKENS", "500")),
            temperature=float(os.getenv("AETHER_LLM_TEMPERATURE", "0.7")),
            max_history_turns=int(os.getenv("AETHER_LLM_MAX_HISTORY_TURNS", "20")),
            base_url=cls.OPENROUTER_BASE_URL,
        )


def _infer_provider_from_model(model: str) -> str:
    """Infer the LLM provider from the model name (for logging/display)."""
    # If already scoped (e.g. "anthropic/claude-3.5-sonnet"), extract prefix.
    if "/" in model:
        return model.split("/", 1)[0].lower()
    lowered = model.strip().lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("deepseek"):
        return "deepseek"
    if lowered.startswith(("gemini", "gemma")):
        return "google"
    if lowered.startswith("glm"):
        return "z-ai"
    if lowered.startswith("llama"):
        return "meta"
    if lowered.startswith("minimax"):
        return "minimax"
    return "openai"


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

    @classmethod
    def from_env(cls) -> TTSConfig:
        return cls(
            provider=os.getenv("AETHER_TTS_PROVIDER", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("AETHER_TTS_MODEL", "tts-1"),
            voice=os.getenv("AETHER_TTS_VOICE", "nova"),
            timeout=float(os.getenv("AETHER_TTS_TIMEOUT", "15.0")),
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

    mode: str = "active"  # off | shadow | active
    model_path: str = ""
    sample_rate: int = 16000
    activation_threshold: float = 0.5
    deactivation_threshold: float = 0.35
    min_speech_duration: float = 0.05
    min_silence_duration: float = 0.55

    @classmethod
    def from_env(cls) -> "VADConfig":
        return cls(
            mode=os.getenv("AETHER_VAD_MODE", "active").lower(),
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
class VoiceBackendConfig:
    """Realtime voice backend settings."""

    backend: str = "gemini"
    api_key: str = ""
    realtime_model: str = "gemini-2.5-flash-preview-native-audio-dialog"
    text_model: str = "gemini-2.5-flash"
    voice: str = "Puck"
    language: str = "en"
    temperature: float = 0.7
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    enable_session_resumption: bool = True

    @classmethod
    def from_env(cls) -> "VoiceBackendConfig":
        return cls(
            backend=os.getenv("AETHER_VOICE_BACKEND", "gemini"),
            api_key=os.getenv("GEMINI_API_KEY", ""),
            realtime_model=os.getenv(
                "AETHER_GEMINI_REALTIME_MODEL",
                "gemini-2.5-flash-preview-native-audio-dialog",
            ),
            text_model=os.getenv("AETHER_GEMINI_TEXT_MODEL", "gemini-2.5-flash"),
            voice=os.getenv("AETHER_GEMINI_VOICE", "Puck"),
            language=os.getenv("AETHER_GEMINI_LANGUAGE", "en"),
            temperature=float(os.getenv("AETHER_GEMINI_TEMPERATURE", "0.7")),
            input_sample_rate=int(
                os.getenv("AETHER_GEMINI_INPUT_SAMPLE_RATE", "16000")
            ),
            output_sample_rate=int(
                os.getenv("AETHER_GEMINI_OUTPUT_SAMPLE_RATE", "24000")
            ),
            enable_session_resumption=os.getenv(
                "AETHER_GEMINI_ENABLE_SESSION_RESUMPTION", "true"
            ).lower()
            == "true",
        )


@dataclass(frozen=True)
class AetherConfig:
    """Root configuration — one object to rule them all."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    webrtc: WebRTCConfig = field(default_factory=WebRTCConfig)
    telephony: TelephonyConfig = field(default_factory=TelephonyConfig)
    voice_backend: VoiceBackendConfig = field(default_factory=VoiceBackendConfig)

    @classmethod
    def from_env(cls) -> AetherConfig:
        return cls(
            llm=LLMConfig.from_env(),
            tts=TTSConfig.from_env(),
            memory=MemoryConfig.from_env(),
            personality=PersonalityConfig.from_env(),
            server=ServerConfig.from_env(),
            kernel=KernelConfig.from_env(),
            vad=VADConfig.from_env(),
            webrtc=WebRTCConfig.from_env(),
            telephony=TelephonyConfig.from_env(),
            voice_backend=VoiceBackendConfig.from_env(),
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
