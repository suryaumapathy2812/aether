"""
Simplified WebRTC transport — signaling + audio routing.

Creates a VoiceSession per peer connection. No CoreMsg, no TransportManager.
Direct: audio frames → VoiceSession → AgentCore → TTS → audio out.

Requires aiortc (optional dependency). If not installed, the transport
is not available and main.py skips it gracefully.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aether.voice.session import VoiceSession

if TYPE_CHECKING:
    from aether.agent import AgentCore
    from aether.providers.base import TTSProvider

logger = logging.getLogger(__name__)

# Optional aiortc imports — graceful degradation if not installed
try:
    import numpy as np
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.contrib.media import MediaStreamTrack
    from av import AudioFrame

    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    np = None  # type: ignore[assignment]


@dataclass
class WebRTCConnection:
    """State for one WebRTC peer connection."""

    pc_id: str
    pc: Any  # RTCPeerConnection
    voice_session: VoiceSession
    audio_out_track: Any  # AudioOutputTrack
    data_channel: Any | None = None


class WebRTCVoiceTransport:
    """
    Simplified WebRTC transport.

    Creates a VoiceSession per connection. Handles:
    - SDP offer/answer exchange
    - ICE candidate trickle
    - Audio frame routing (inbound mic → STT, outbound TTS → speaker)
    - Data channel for text events (transcript, status, tool_result)
    """

    def __init__(
        self,
        agent: "AgentCore",
        tts_provider: "TTSProvider",
        ice_servers: list[dict] | None = None,
    ) -> None:
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc is required for WebRTC transport")

        self.agent = agent
        self.tts_provider = tts_provider
        self.ice_servers = ice_servers or [{"urls": "stun:stun.l.google.com:19302"}]
        self._connections: dict[str, WebRTCConnection] = {}

    async def handle_offer(
        self,
        sdp: str,
        sdp_type: str = "offer",
        user_id: str = "",
        pc_id: str | None = None,
    ) -> dict:
        """Handle SDP offer → create peer connection + VoiceSession → return answer."""

        pc_id = pc_id or f"pc-{uuid.uuid4().hex[:12]}"

        # Build ICE configuration
        ice_list = []
        for server in self.ice_servers:
            urls = server.get("urls", "")
            if server.get("username"):
                ice_list.append(
                    RTCIceServer(
                        urls=urls,
                        username=server["username"],
                        credential=server.get("credential", ""),
                    )
                )
            else:
                ice_list.append(RTCIceServer(urls=urls))

        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_list))

        # Create voice session
        voice_session = VoiceSession(
            agent=self.agent,
            tts_provider=self.tts_provider,
            session_id=f"webrtc-{pc_id}",
        )

        # Create outbound audio track
        audio_out = AudioOutputTrack()
        pc.addTrack(audio_out)

        # Wire voice session audio output → outbound track
        async def send_audio(audio_bytes: bytes) -> None:
            audio_out.add_audio(audio_bytes)

        voice_session.on_audio_out = send_audio

        # Store connection
        conn = WebRTCConnection(
            pc_id=pc_id,
            pc=pc,
            voice_session=voice_session,
            audio_out_track=audio_out,
        )
        self._connections[pc_id] = conn

        # Setup event handlers
        self._setup_handlers(conn)

        # Set remote description and create answer
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info("WebRTC offer handled: pc_id=%s", pc_id)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "pc_id": pc_id,
        }

    async def handle_ice_candidate(
        self,
        pc_id: str,
        candidate: str,
        sdp_mid: str | None = None,
        sdp_mline_index: int | None = None,
    ) -> None:
        """Add an ICE candidate to a peer connection."""
        conn = self._connections.get(pc_id)
        if not conn:
            raise ValueError(f"Unknown pc_id: {pc_id}")

        from aiortc import RTCIceCandidate

        # Parse candidate string — aiortc expects specific format
        if candidate:
            await conn.pc.addIceCandidate(
                RTCIceCandidate(
                    sdpMid=sdp_mid,
                    sdpMLineIndex=sdp_mline_index,
                    candidate=candidate,
                )
            )

    def _setup_handlers(self, conn: WebRTCConnection) -> None:
        """Wire up WebRTC event handlers for a connection."""
        pc = conn.pc

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            state = pc.connectionState
            logger.info("WebRTC %s state: %s", conn.pc_id, state)

            if state in ("disconnected", "failed", "closed"):
                await conn.voice_session.stop()
                self._connections.pop(conn.pc_id, None)

        @pc.on("track")
        async def on_track(track: Any) -> None:
            if track.kind == "audio":
                asyncio.create_task(
                    self._read_audio_loop(conn, track),
                    name=f"audio-in-{conn.pc_id}",
                )

        @pc.on("datachannel")
        async def on_datachannel(channel: Any) -> None:
            conn.data_channel = channel

            # Wire text events from VoiceSession → data channel
            async def send_event(event: dict) -> None:
                if conn.data_channel and conn.data_channel.readyState == "open":
                    conn.data_channel.send(json.dumps(event))

            conn.voice_session.on_text_event = send_event

            @channel.on("message")
            async def on_message(message: str) -> None:
                await self._handle_data_message(conn, message)

    async def _read_audio_loop(self, conn: WebRTCConnection, track: Any) -> None:
        """Read audio frames from WebRTC inbound track → VoiceSession STT."""
        try:
            while True:
                frame = await track.recv()
                if not conn.voice_session.is_streaming:
                    continue  # Drop until stream_start

                # Convert to PCM16 mono at 16kHz for STT
                pcm = frame.to_ndarray().flatten()

                # Resample if needed (WebRTC typically sends 48kHz)
                if frame.sample_rate != 16000:
                    pcm = _resample(pcm, frame.sample_rate, 16000)

                pcm_bytes = (pcm * 32767).astype(np.int16).tobytes()
                await conn.voice_session.on_audio_in(pcm_bytes)

        except Exception:
            pass  # Connection closed

    async def _handle_data_message(self, conn: WebRTCConnection, message: str) -> None:
        """Handle data channel messages from the client."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "stream_start":
                await conn.voice_session.start()

            elif msg_type == "stream_stop":
                await conn.voice_session.stop()

            elif msg_type == "mute":
                conn.voice_session.mute()

            elif msg_type == "unmute":
                conn.voice_session.unmute()

            elif msg_type == "text":
                # Text input via data channel (rare but supported)
                text = data.get("data", "")
                if text:
                    await conn.voice_session._trigger_response(text)

        except json.JSONDecodeError:
            pass

    async def close_all(self) -> None:
        """Close all connections (shutdown)."""
        for conn in list(self._connections.values()):
            try:
                await conn.voice_session.stop()
                await conn.pc.close()
            except Exception:
                pass
        self._connections.clear()


# ─── Audio Output Track ──────────────────────────────────────────


class AudioOutputTrack(MediaStreamTrack):
    """
    Outbound audio track that plays TTS audio to the WebRTC peer.

    TTS audio bytes are queued and served as 20ms frames at 48kHz
    (standard WebRTC audio frame size).
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._sample_rate = 48000
        self._samples_per_frame = 960  # 20ms at 48kHz
        self._buffer = b""
        self._pts = 0

    def add_audio(self, audio_bytes: bytes) -> None:
        """Queue TTS audio bytes for playback."""
        self._queue.put_nowait(audio_bytes)

    async def recv(self) -> "AudioFrame":
        """Called by aiortc to get the next audio frame."""
        frame_size = self._samples_per_frame * 2  # 16-bit = 2 bytes per sample

        # Fill buffer from queue
        while len(self._buffer) < frame_size:
            try:
                chunk = self._queue.get_nowait()
                self._buffer += chunk
            except asyncio.QueueEmpty:
                # No audio available — send silence
                self._buffer += b"\x00" * frame_size
                break

        # Extract one frame
        frame_bytes = self._buffer[:frame_size]
        self._buffer = self._buffer[frame_size:]

        # Build AudioFrame
        frame = AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        self._pts += self._samples_per_frame

        # Copy PCM data into frame
        frame.planes[0].update(frame_bytes)

        return frame


# ─── Audio Helpers ───────────────────────────────────────────────


def _resample(pcm: "np.ndarray", src_rate: int, dst_rate: int) -> "np.ndarray":
    """Simple linear resampling. For production, use scipy or soxr."""
    if src_rate == dst_rate:
        return pcm
    ratio = dst_rate / src_rate
    n_samples = int(len(pcm) * ratio)
    indices = np.linspace(0, len(pcm) - 1, n_samples)
    return np.interp(indices, np.arange(len(pcm)), pcm)
