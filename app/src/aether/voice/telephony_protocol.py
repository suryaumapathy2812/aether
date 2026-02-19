"""
Telephony protocol adapters — provider-specific WebSocket media stream handling.

Supports Twilio, Telnyx, and Vobiz media stream protocols.
Each adapter normalizes the provider's WebSocket messages into a common format.
"""

from __future__ import annotations

import audioop
import base64
import json
import logging
import struct
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class TelephonyProtocol(ABC):
    """Base class for telephony provider protocol adapters."""

    @abstractmethod
    def parse_media_message(self, message: str) -> dict[str, Any] | None:
        """Parse a WebSocket message and return normalized audio data.

        Returns dict with:
            - type: "media" | "start" | "stop" | "mark"
            - audio: bytes (raw mulaw for media type)
            - stream_sid: str
            - sequence: int
        Or None if the message should be ignored.
        """
        ...

    @abstractmethod
    def encode_audio_message(
        self,
        audio_bytes: bytes,
        stream_sid: str,
        *,
        content_type: str = "audio/x-mulaw",
        sample_rate: int = 8000,
    ) -> str:
        """Encode audio bytes into a provider-specific WebSocket message."""
        ...

    @abstractmethod
    def create_mark_message(self, stream_sid: str, name: str) -> str:
        """Create a mark message for playback synchronization."""
        ...

    def create_clear_message(self, stream_sid: str) -> str | None:
        """Create provider-specific clear-buffer message if supported."""
        return None

    def create_checkpoint_message(self, stream_sid: str, name: str) -> str | None:
        """Create provider-specific checkpoint message if supported."""
        return None


class TwilioProtocol(TelephonyProtocol):
    """Twilio Media Streams protocol adapter.

    Twilio sends mulaw 8kHz audio as base64-encoded payloads in JSON messages.
    """

    def parse_media_message(self, message: str) -> dict[str, Any] | None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return None

        event = data.get("event", "")

        if event == "media":
            media = data.get("media", {})
            audio_b64 = media.get("payload", "")
            if not audio_b64:
                return None
            return {
                "type": "media",
                "audio": base64.b64decode(audio_b64),
                "stream_sid": data.get("streamSid", ""),
                "sequence": int(media.get("chunk", 0)),
            }

        elif event == "start":
            start = data.get("start", {})
            return {
                "type": "start",
                "stream_sid": data.get("streamSid", ""),
                "call_sid": start.get("callSid", ""),
                "from": start.get("customParameters", {}).get("from", ""),
                "to": start.get("customParameters", {}).get("to", ""),
            }

        elif event == "stop":
            return {
                "type": "stop",
                "stream_sid": data.get("streamSid", ""),
            }

        elif event == "mark":
            return {
                "type": "mark",
                "stream_sid": data.get("streamSid", ""),
                "name": data.get("mark", {}).get("name", ""),
            }

        return None

    def encode_audio_message(
        self,
        audio_bytes: bytes,
        stream_sid: str,
        *,
        content_type: str = "audio/x-mulaw",
        sample_rate: int = 8000,
    ) -> str:
        payload = base64.b64encode(audio_bytes).decode("ascii")
        return json.dumps(
            {
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload},
            }
        )

    def create_mark_message(self, stream_sid: str, name: str) -> str:
        return json.dumps(
            {
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {"name": name},
            }
        )


class TelnyxProtocol(TelephonyProtocol):
    """Telnyx Media Streams protocol adapter.

    Similar to Twilio but with slightly different message format.
    """

    def parse_media_message(self, message: str) -> dict[str, Any] | None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return None

        event = data.get("event", "")

        if event == "media":
            media = data.get("media", {})
            audio_b64 = media.get("payload", "")
            if not audio_b64:
                return None
            return {
                "type": "media",
                "audio": base64.b64decode(audio_b64),
                "stream_sid": data.get("stream_id", ""),
                "sequence": int(media.get("chunk", 0)),
            }

        elif event == "start":
            return {
                "type": "start",
                "stream_sid": data.get("stream_id", ""),
                "call_control_id": data.get("start", {}).get("call_control_id", ""),
            }

        elif event == "stop":
            return {
                "type": "stop",
                "stream_sid": data.get("stream_id", ""),
            }

        return None

    def encode_audio_message(
        self,
        audio_bytes: bytes,
        stream_sid: str,
        *,
        content_type: str = "audio/x-mulaw",
        sample_rate: int = 8000,
    ) -> str:
        payload = base64.b64encode(audio_bytes).decode("ascii")
        return json.dumps(
            {
                "event": "media",
                "stream_id": stream_sid,
                "media": {"payload": payload},
            }
        )

    def create_mark_message(self, stream_sid: str, name: str) -> str:
        return json.dumps(
            {
                "event": "mark",
                "stream_id": stream_sid,
                "mark": {"name": name},
            }
        )


class VobizProtocol(TelephonyProtocol):
    """Vobiz XML Stream WebSocket protocol adapter."""

    def parse_media_message(self, message: str) -> dict[str, Any] | None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return None

        event = data.get("event", "")

        if event == "start":
            start = data.get("start", {})
            media_format = start.get("mediaFormat", {})
            return {
                "type": "start",
                "stream_sid": start.get("streamId", data.get("streamId", "")),
                "call_id": start.get("callId", ""),
                "sequence": int(data.get("sequenceNumber", 0)),
                "content_type": str(media_format.get("encoding", "audio/x-l16")),
                "sample_rate": int(media_format.get("sampleRate", 8000)),
            }

        if event == "media":
            media = data.get("media", {})
            audio_b64 = media.get("payload", "")
            if not audio_b64:
                return None
            return {
                "type": "media",
                "audio": base64.b64decode(audio_b64),
                "stream_sid": data.get("streamId", ""),
                "sequence": int(data.get("sequenceNumber", 0)),
                "content_type": str(media.get("contentType", "audio/x-l16")),
                "sample_rate": int(media.get("sampleRate", 8000)),
            }

        if event == "stop":
            return {
                "type": "stop",
                "stream_sid": data.get("streamId", ""),
                "reason": data.get("reason", ""),
            }

        if event == "playedStream":
            return {
                "type": "checkpoint_ack",
                "stream_sid": data.get("streamId", ""),
                "name": data.get("name", ""),
            }

        if event == "clearedAudio":
            return {
                "type": "clear_ack",
                "stream_sid": data.get("streamId", ""),
            }

        return None

    def encode_audio_message(
        self,
        audio_bytes: bytes,
        stream_sid: str,
        *,
        content_type: str = "audio/x-mulaw",
        sample_rate: int = 8000,
    ) -> str:
        payload = base64.b64encode(audio_bytes).decode("ascii")
        return json.dumps(
            {
                "event": "playAudio",
                "media": {
                    "contentType": content_type,
                    "sampleRate": sample_rate,
                    "payload": payload,
                },
            }
        )

    def create_mark_message(self, stream_sid: str, name: str) -> str:
        return self.create_checkpoint_message(stream_sid, name) or ""

    def create_clear_message(self, stream_sid: str) -> str:
        return json.dumps({"event": "clearAudio", "streamId": stream_sid})

    def create_checkpoint_message(self, stream_sid: str, name: str) -> str:
        return json.dumps(
            {
                "event": "checkpoint",
                "streamId": stream_sid,
                "name": name,
            }
        )


# ─── Audio Codec Helpers ─────────────────────────────────────────


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Decode mulaw to 16-bit PCM."""
    return audioop.ulaw2lin(mulaw_bytes, 2)


def pcm16_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Encode 16-bit PCM to mulaw."""
    return audioop.lin2ulaw(pcm_bytes, 2)


def resample_8k_to_16k(pcm_8k: bytes) -> bytes:
    """Resample 8kHz PCM16 to 16kHz PCM16 using linear interpolation."""
    return audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)[0]


def resample_24k_to_8k(pcm_24k: bytes) -> bytes:
    """Resample 24kHz PCM16 to 8kHz PCM16."""
    return audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)[0]


def resample_24k_to_16k(pcm_24k: bytes) -> bytes:
    """Resample 24kHz PCM16 to 16kHz PCM16."""
    return audioop.ratecv(pcm_24k, 2, 1, 24000, 16000, None)[0]


def get_protocol(provider: str) -> TelephonyProtocol:
    """Get the protocol adapter for a telephony provider."""
    protocols = {
        "twilio": TwilioProtocol,
        "telnyx": TelnyxProtocol,
        "vobiz": VobizProtocol,
    }
    proto_class = protocols.get(provider)
    if not proto_class:
        raise ValueError(f"Unknown telephony provider: {provider}")
    return proto_class()
