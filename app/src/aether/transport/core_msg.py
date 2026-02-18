"""
Core Message Types - The unified message format between transport and core.

This module defines the CoreMsg format that all transports use to communicate
with the Aether Core. The Core doesn't know about transport specifics.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Union


class MsgDirection(str, Enum):
    """Direction of message flow."""

    INBOUND = "inbound"  # Client → Core
    OUTBOUND = "outbound"  # Core → Client


class ConnectionState(str, Enum):
    """Connection state for tracking client connections."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTED = "reconnected"


@dataclass
class TextContent:
    """Text content in a message."""

    text: str
    role: str = "user"  # "user" | "assistant"


@dataclass
class AudioContent:
    """Audio content in a message."""

    audio_data: bytes
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class EventContent:
    """Event content in a message (for plugin events, etc.)."""

    event_type: str
    payload: dict = field(default_factory=dict)


@dataclass
class MsgMetadata:
    """Metadata about the message."""

    transport: str = "unknown"  # "websocket", "webrtc", "rest", "push"
    client_info: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    session_mode: str = "voice"  # "voice" | "text"
    sentence_index: int = 0  # ordering for text_chunk / audio_chunk


@dataclass
class CoreMsg:
    """
    Unified message format between transport and core.

    This is the standard format that all transports use to communicate
    with the Aether Core. The Core only speaks in CoreMsg, not knowing
    about WebSocket, WebRTC, or any transport specifics.
    """

    # Identity
    user_id: str
    session_id: str

    # Content
    content: Union[TextContent, AudioContent, EventContent]

    # Metadata
    metadata: MsgMetadata = field(default_factory=MsgMetadata)

    # Direction
    direction: MsgDirection = MsgDirection.INBOUND

    # Unique ID for tracking
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Response callback (for streaming responses)
    _response_callback: Callable[["CoreMsg"], Any] = field(default=None, repr=False)

    def response(self, msg: "CoreMsg") -> None:
        """Send a response back through the transport."""
        if self._response_callback:
            self._response_callback(msg)

    @classmethod
    def text(
        cls,
        text: str,
        user_id: str,
        session_id: str,
        role: str = "user",
        transport: str = "websocket",
        **kwargs,
    ) -> "CoreMsg":
        """Create a text message."""
        return cls(
            user_id=user_id,
            session_id=session_id,
            content=TextContent(text=text, role=role),
            metadata=MsgMetadata(transport=transport, **kwargs),
            direction=MsgDirection.INBOUND if role == "user" else MsgDirection.OUTBOUND,
        )

    @classmethod
    def audio(
        cls,
        audio_data: bytes,
        user_id: str,
        session_id: str,
        sample_rate: int = 16000,
        channels: int = 1,
        transport: str = "websocket",
        **kwargs,
    ) -> "CoreMsg":
        """Create an audio message."""
        return cls(
            user_id=user_id,
            session_id=session_id,
            content=AudioContent(
                audio_data=audio_data, sample_rate=sample_rate, channels=channels
            ),
            metadata=MsgMetadata(transport=transport, **kwargs),
            direction=MsgDirection.INBOUND,
        )

    @classmethod
    def event(
        cls,
        event_type: str,
        user_id: str,
        session_id: str,
        payload: dict = None,
        transport: str = "rest",
        **kwargs,
    ) -> "CoreMsg":
        """Create an event message."""
        return cls(
            user_id=user_id,
            session_id=session_id,
            content=EventContent(event_type=event_type, payload=payload or {}),
            metadata=MsgMetadata(transport=transport, **kwargs),
            direction=MsgDirection.INBOUND,
        )
