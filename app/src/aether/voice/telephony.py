"""
Telephony Transport — phone calls via WebSocket media streams.

Reuses VoiceSession entirely. Only the audio I/O layer changes:
- WebRTC: 48kHz PCM via RTCPeerConnection
- Telephony: 8kHz mulaw via WebSocket

Audio pipeline:
  Receive: 8kHz mulaw → decode → resample to 16kHz → VAD → VoiceSession.on_audio_in()
  Send:    TTS 24kHz PCM → resample to 8kHz → mulaw encode → WebSocket → phone

Supports Twilio, Telnyx, and Vobiz via protocol adapters.
Outbound calls are handled by provider-specific plugins (e.g., Vobiz plugin).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

from aether.core.config import config
from aether.voice.session import VoiceSession
from aether.voice.telephony_protocol import (
    TelephonyProtocol,
    get_protocol,
    mulaw_to_pcm16,
    pcm16_to_mulaw,
    resample_24k_to_16k,
    resample_8k_to_16k,
    resample_24k_to_8k,
)
from aether.voice.vad import VADSettings, build_vad

if TYPE_CHECKING:
    from fastapi import WebSocket

    from aether.agent import AgentCore
    from aether.providers.base import TTSProvider

logger = logging.getLogger(__name__)


class TelephonySession:
    """State for one phone call WebSocket connection."""

    def __init__(
        self,
        call_id: str,
        stream_sid: str,
        voice_session: VoiceSession,
        protocol: TelephonyProtocol,
        ws: "WebSocket",
        vad: Any | None = None,
        vad_mode: str = "off",
    ) -> None:
        self.call_id = call_id
        self.stream_sid = stream_sid
        self.voice_session = voice_session
        self.protocol = protocol
        self.ws = ws
        self.vad = vad
        self.vad_mode = vad_mode
        self.connected_at = time.time()
        self._audio_chunks_received = 0
        self._audio_chunks_sent = 0
        self.input_content_type = "audio/x-mulaw"
        self.input_sample_rate = 8000
        self.output_content_type = "audio/x-mulaw"
        self.output_sample_rate = 8000


class TelephonyTransport:
    """
    Telephony transport — phone calls via WebSocket media streams.

    Creates a VoiceSession per phone call. The session is identical to
    WebRTC sessions — same LLM, memory, tools, plugins, turn detection.
    Only the audio codec and transport differ.
    """

    def __init__(
        self,
        agent: "AgentCore",
        tts_provider: "TTSProvider",
    ) -> None:
        self.agent = agent
        self.tts_provider = tts_provider
        self._sessions: dict[str, TelephonySession] = {}
        self.on_sessions_empty: Callable[[], Any] | None = None

    async def handle_call(self, ws: "WebSocket") -> None:
        """Handle a phone call WebSocket connection.

        This is the main entry point — called from the FastAPI WebSocket endpoint.
        Runs for the duration of the call.
        """
        await ws.accept()

        # Default to Vobiz protocol (most common for this plugin)
        # The protocol can be passed in or determined from the first message
        protocol = get_protocol("vobiz")
        call_id = f"tel-{uuid.uuid4().hex[:12]}"
        session: TelephonySession | None = None

        try:
            async for raw_message in ws.iter_text():
                parsed = protocol.parse_media_message(raw_message)
                if not parsed:
                    continue

                msg_type = parsed["type"]

                if msg_type == "start":
                    # Call started — create VoiceSession
                    stream_sid = parsed["stream_sid"]
                    logger.info(
                        "Telephony call started: call_id=%s stream_sid=%s",
                        call_id,
                        stream_sid,
                    )

                    voice_session = VoiceSession(
                        agent=self.agent,
                        tts_provider=self.tts_provider,
                        session_id=call_id,
                    )

                    # Build VAD
                    vad = None
                    vad_mode = config.vad.mode
                    if vad_mode in ("shadow", "active"):
                        vad = build_vad(
                            VADSettings(
                                mode=vad_mode,
                                model_path=config.vad.model_path,
                                sample_rate=config.vad.sample_rate,
                                activation_threshold=config.vad.activation_threshold,
                                deactivation_threshold=config.vad.deactivation_threshold,
                                min_speech_duration=config.vad.min_speech_duration,
                                min_silence_duration=config.vad.min_silence_duration,
                            )
                        )

                    session = TelephonySession(
                        call_id=call_id,
                        stream_sid=stream_sid,
                        voice_session=voice_session,
                        protocol=protocol,
                        ws=ws,
                        vad=vad,
                        vad_mode=vad_mode,
                    )
                    self._sessions[call_id] = session

                    session.input_content_type = str(
                        parsed.get("content_type", "audio/x-mulaw")
                    ).lower()
                    session.input_sample_rate = int(parsed.get("sample_rate", 8000))

                    session.output_content_type, session.output_sample_rate = (
                        _select_output_audio_format()
                    )

                    # Wire audio output: TTS 24kHz PCM → 8kHz mulaw → WebSocket
                    async def send_audio(
                        audio_bytes: bytes,
                        _session: TelephonySession = session,
                    ) -> None:
                        try:
                            # TTS produces 24kHz PCM16 mono
                            out_bytes, chunk_size = _encode_outbound_audio(
                                audio_bytes,
                                content_type=_session.output_content_type,
                                sample_rate=_session.output_sample_rate,
                            )

                            for i in range(0, len(out_bytes), chunk_size):
                                chunk = out_bytes[i : i + chunk_size]
                                msg = _session.protocol.encode_audio_message(
                                    chunk,
                                    _session.stream_sid,
                                    content_type=_session.output_content_type,
                                    sample_rate=_session.output_sample_rate,
                                )
                                await _session.ws.send_text(msg)
                                _session._audio_chunks_sent += 1
                        except Exception as e:
                            logger.warning("Telephony send_audio error: %s", e)

                    async def on_barge_in(
                        _session: TelephonySession = session,
                    ) -> None:
                        clear_msg = _session.protocol.create_clear_message(
                            _session.stream_sid
                        )
                        if clear_msg:
                            try:
                                await _session.ws.send_text(clear_msg)
                            except Exception as e:
                                logger.debug("Telephony clearAudio send failed: %s", e)
                        logger.info("Telephony barge-in on call %s", _session.call_id)

                    def on_tts_duration(
                        playback_secs: float,
                        _session: TelephonySession = session,
                    ) -> None:
                        now = time.time()
                        current_end = max(
                            now,
                            _session.voice_session._assistant_speaking_until,
                        )
                        _session.voice_session._assistant_speaking_until = (
                            current_end + playback_secs
                        )
                        _session.voice_session._tts_cooldown_until = (
                            _session.voice_session._assistant_speaking_until + 0.3
                        )

                    async def send_event(
                        event: dict,
                        _session: TelephonySession = session,
                    ) -> None:
                        # Text events are not sent over phone — log only
                        logger.debug(
                            "Telephony text event (call=%s): %s",
                            _session.call_id,
                            event.get("type", ""),
                        )

                    voice_session.on_audio_out = send_audio
                    voice_session.on_barge_in = on_barge_in
                    voice_session.on_tts_duration = on_tts_duration
                    voice_session.on_text_event = send_event

                    # Start the voice session
                    await voice_session.start()

                elif msg_type == "media" and session:
                    # Incoming audio from phone
                    session._audio_chunks_received += 1
                    pcm_16k = _decode_inbound_audio(
                        parsed["audio"],
                        content_type=str(
                            parsed.get("content_type", session.input_content_type)
                        ).lower(),
                        sample_rate=int(
                            parsed.get("sample_rate", session.input_sample_rate)
                        ),
                    )

                    # Run VAD
                    if session.vad is not None:
                        for event in session.vad.process_pcm16(pcm_16k):
                            await session.voice_session.on_vad_event(
                                event, session.vad_mode
                            )

                    # Send to voice session
                    await session.voice_session.on_audio_in(pcm_16k)

                elif msg_type == "stop" and session:
                    logger.info("Telephony call ended: %s", call_id)
                    break

                elif msg_type in {"checkpoint_ack", "clear_ack"} and session:
                    logger.debug(
                        "Telephony control ack: %s call=%s",
                        msg_type,
                        session.call_id,
                    )

        except Exception as e:
            logger.error("Telephony WebSocket error: %s", e, exc_info=True)
        finally:
            # Clean up
            if session:
                await session.voice_session.stop()
                self._sessions.pop(call_id, None)
                logger.info(
                    "Telephony call cleaned up: %s (duration=%.1fs, in=%d out=%d)",
                    call_id,
                    time.time() - session.connected_at,
                    session._audio_chunks_received,
                    session._audio_chunks_sent,
                )

                if not self._sessions and self.on_sessions_empty:
                    asyncio.create_task(self.on_sessions_empty())

    def get_active_sessions(self) -> list[VoiceSession]:
        """Return all active voice sessions from phone calls."""
        return [
            s.voice_session
            for s in self._sessions.values()
            if s.voice_session.is_streaming
        ]


def _decode_inbound_audio(
    audio_bytes: bytes,
    *,
    content_type: str,
    sample_rate: int,
) -> bytes:
    """Normalize telephony inbound audio to 16kHz PCM16 mono."""
    if "mulaw" in content_type:
        pcm = mulaw_to_pcm16(audio_bytes)
    else:
        # audio/x-l16 is raw PCM16 mono
        pcm = audio_bytes

    if sample_rate == 16000:
        return pcm
    if sample_rate == 8000:
        return resample_8k_to_16k(pcm)

    # Fallback: treat as 8k if provider sends unknown rate.
    return resample_8k_to_16k(pcm)


def _encode_outbound_audio(
    pcm_24k: bytes,
    *,
    content_type: str,
    sample_rate: int,
) -> tuple[bytes, int]:
    """Encode outbound telephony audio and return (bytes, frame_chunk_size)."""
    if "mulaw" in content_type:
        # mu-law in Vobiz/Twilio is 8kHz only.
        pcm_8k = resample_24k_to_8k(pcm_24k)
        return pcm16_to_mulaw(pcm_8k), 160  # 20ms @ 8k mulaw

    # Linear PCM path (audio/x-l16)
    if sample_rate == 16000:
        return resample_24k_to_16k(pcm_24k), 640  # 20ms @ 16k PCM16 mono
    return resample_24k_to_8k(pcm_24k), 320  # 20ms @ 8k PCM16 mono


def _select_output_audio_format() -> tuple[str, int]:
    """Map config telephony encoding/sample rate to provider payload format."""
    encoding = config.telephony.encoding.lower()
    sample_rate = config.telephony.sample_rate

    if encoding in {"mulaw", "mu-law", "ulaw", "audio/x-mulaw"}:
        return "audio/x-mulaw", 8000
    if encoding in {"l16", "pcm", "audio/x-l16"}:
        if sample_rate not in {8000, 16000}:
            sample_rate = 8000
        return "audio/x-l16", sample_rate

    return "audio/x-mulaw", 8000
