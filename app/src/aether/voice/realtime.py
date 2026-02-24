"""
Backend-agnostic realtime voice model interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class RealtimeEventType(Enum):
    """Events emitted by a RealtimeSession."""

    AUDIO_DELTA = "audio_delta"
    AUDIO_DONE = "audio_done"
    TEXT_DELTA = "text_delta"
    TEXT_DONE = "text_done"
    INPUT_SPEECH_STARTED = "input_speech_started"
    INPUT_SPEECH_STOPPED = "input_speech_stopped"
    INPUT_SPEECH_COMMITTED = "input_speech_committed"
    INPUT_SPEECH_TRANSCRIPTION_COMPLETED = "input_speech_transcription_completed"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_DONE = "function_call_done"
    SESSION_CREATED = "session_created"
    SESSION_RESUMED = "session_resumed"
    SESSION_ERROR = "session_error"
    TURN_STARTED = "turn_started"
    TURN_DONE = "turn_done"
    INTERRUPTED = "interrupted"


@dataclass
class RealtimeEvent:
    """A model event consumed by VoiceSession."""

    type: RealtimeEventType
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionCallResult:
    """Tool/function execution result sent back to the realtime model."""

    call_id: str
    name: str
    result: str


@dataclass
class RealtimeModelConfig:
    """Config used to construct concrete realtime model implementations."""

    model: str = ""
    api_key: str = ""
    voice: str = "Puck"
    instructions: str = ""
    temperature: float = 0.7
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    language: str = "en"
    response_modalities: list[str] = field(default_factory=lambda: ["AUDIO"])
    enable_session_resumption: bool = True


class RealtimeSession(ABC):
    """Live session connection to a realtime model backend."""

    @abstractmethod
    async def push_audio(self, data: bytes) -> None: ...

    @abstractmethod
    def events(self) -> AsyncIterator[RealtimeEvent]: ...

    @abstractmethod
    async def send_function_result(self, result: FunctionCallResult) -> None: ...

    @abstractmethod
    async def interrupt(self) -> None: ...

    @abstractmethod
    async def generate_reply(self, text: str) -> None: ...

    @abstractmethod
    async def update_instructions(self, instructions: str) -> None: ...

    @abstractmethod
    async def update_tools(self, tools: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def session_id(self) -> str: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @property
    @abstractmethod
    def resumption_token(self) -> str | None: ...


class RealtimeModel(ABC):
    """Factory for creating realtime sessions."""

    @abstractmethod
    async def create_session(
        self,
        *,
        instructions: str = "",
        tools: list[dict[str, Any]] | None = None,
        resumption_token: str | None = None,
    ) -> RealtimeSession: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def config(self) -> RealtimeModelConfig: ...
