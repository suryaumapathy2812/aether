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
from typing import TYPE_CHECKING, Any, Callable

from aether.core.config import config
from aether.voice.session import VoiceSession
from aether.voice.vad import VADSettings, build_vad

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
    vad: Any | None = None
    vad_mode: str = "off"
    _audio_frame_count: int = field(default=0, repr=False)
    is_reconnect: bool = False  # True if reusing an existing VoiceSession
    session_key: str = ""  # Key into WebRTCVoiceTransport._sessions


class WebRTCVoiceTransport:
    """
    Simplified WebRTC transport with persistent session support.

    Creates a VoiceSession per user (not per connection). On disconnect,
    the VoiceSession is paused but kept alive — reconnecting reuses it
    instantly without losing dialog history or agent state.

    Handles:
    - SDP offer/answer exchange
    - ICE candidate trickle
    - Audio frame routing (inbound mic → STT, outbound TTS → speaker)
    - Data channel for text events (transcript, status, tool_result)
    - Pause/resume on disconnect/reconnect (persistent session)
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
        # Persistent sessions keyed by user_id — survive WebRTC disconnects
        self._sessions: dict[str, VoiceSession] = {}
        # Optional callback: called with False when all persistent sessions are gone.
        # Wire this up in main.py to clear the orchestrator keep_alive flag.
        self.on_sessions_empty: Callable[[], Any] | None = None

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

        # Reuse existing session for this user if one exists (persistent session).
        # This preserves dialog history and agent state across reconnects.
        session_key = user_id or pc_id
        is_reconnect = session_key in self._sessions
        if is_reconnect:
            voice_session = self._sessions[session_key]
            logger.info(
                "WebRTC reconnect for %s — reusing existing VoiceSession", session_key
            )
        else:
            voice_session = VoiceSession(
                agent=self.agent,
                tts_provider=self.tts_provider,
                session_id=f"webrtc-{pc_id}",
            )
            self._sessions[session_key] = voice_session

        # Create outbound audio track
        audio_out = AudioOutputTrack()
        pc.addTrack(audio_out)

        # Wire voice session audio output → outbound track
        # Stateful resampler: 24kHz mono s16 → 48kHz mono s16 (matches inbound path)
        from av import AudioResampler as _OutResampler

        tts_resampler = _OutResampler(format="s16", layout="mono", rate=48000)

        async def send_audio(audio_bytes: bytes) -> None:
            # TTS provider returns 24kHz PCM16 mono; WebRTC track expects 48kHz.
            # Build an AudioFrame from the raw PCM so av can resample it properly.
            sample_count = len(audio_bytes) // 2  # 16-bit = 2 bytes per sample
            logger.info(
                "send_audio: %d bytes (%d samples) input at 24kHz",
                len(audio_bytes),
                sample_count,
            )
            src_frame = AudioFrame(format="s16", layout="mono", samples=sample_count)
            src_frame.sample_rate = 24000
            src_frame.planes[0].update(audio_bytes)

            resampled_frames = tts_resampler.resample(src_frame)
            total_queued = 0
            for rf in resampled_frames:
                data = rf.to_ndarray().tobytes()
                total_queued += len(data)
                audio_out.add_audio(data)

            # Calculate true playback duration from resampled PCM:
            # 48kHz mono s16 = 48000 samples/sec × 2 bytes/sample = 96000 bytes/sec
            playback_secs = total_queued / 96000.0
            logger.info(
                "send_audio: resampled to %d bytes (%.2fs), queue size=%d",
                total_queued,
                playback_secs,
                audio_out._queue.qsize(),
            )
            if voice_session.on_tts_duration:
                voice_session.on_tts_duration(playback_secs)

        async def on_barge_in() -> None:
            audio_out.clear_audio()
            logger.info("AudioOutputTrack: cleared buffered audio on barge-in")

        def on_tts_duration(playback_secs: float) -> None:
            """Update echo suppression window with the true PCM playback duration."""
            import time as _time

            now = _time.time()
            voice_session._assistant_speaking_until = now + playback_secs
            voice_session._tts_cooldown_until = now + playback_secs + 0.3

        voice_session.on_audio_out = send_audio
        voice_session.on_barge_in = on_barge_in
        voice_session.on_tts_duration = on_tts_duration

        # Store connection
        conn = WebRTCConnection(
            pc_id=pc_id,
            pc=pc,
            voice_session=voice_session,
            audio_out_track=audio_out,
            vad_mode=config.vad.mode,
            is_reconnect=is_reconnect,
            session_key=session_key,
        )

        if config.vad.mode in ("shadow", "active"):
            conn.vad = build_vad(
                VADSettings(
                    mode=config.vad.mode,
                    model_path=config.vad.model_path,
                    sample_rate=config.vad.sample_rate,
                    activation_threshold=config.vad.activation_threshold,
                    deactivation_threshold=config.vad.deactivation_threshold,
                    min_speech_duration=config.vad.min_speech_duration,
                    min_silence_duration=config.vad.min_silence_duration,
                )
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

        from aiortc.sdp import candidate_from_sdp

        # Parse candidate string — aiortc expects specific format
        if candidate:
            raw = candidate[10:] if candidate.startswith("candidate:") else candidate
            parsed = candidate_from_sdp(raw)
            parsed.sdpMid = sdp_mid
            parsed.sdpMLineIndex = sdp_mline_index
            await conn.pc.addIceCandidate(parsed)

    def _setup_handlers(self, conn: WebRTCConnection) -> None:
        """Wire up WebRTC event handlers for a connection."""
        pc = conn.pc

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            state = pc.connectionState
            logger.info("WebRTC %s state: %s", conn.pc_id, state)

            if state in ("disconnected", "failed", "closed"):
                # Pause (not stop) — keep session alive for fast reconnect.
                # The session's dialog history and agent state are preserved.
                await conn.voice_session.pause()
                self._connections.pop(conn.pc_id, None)
                logger.info(
                    "WebRTC %s disconnected — session paused, ready for reconnect",
                    conn.pc_id,
                )

        @pc.on("track")
        async def on_track(track: Any) -> None:
            if track.kind == "audio":
                asyncio.create_task(
                    self._read_audio_loop(conn, track),
                    name=f"audio-in-{conn.pc_id}",
                )

        @pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            logger.info("Data channel opened: %s", conn.pc_id)
            conn.data_channel = channel

            # Wire text events from VoiceSession → data channel
            async def send_event(event: dict) -> None:
                if conn.data_channel and conn.data_channel.readyState == "open":
                    conn.data_channel.send(json.dumps(event))

            conn.voice_session.on_text_event = send_event

            @channel.on("message")
            def on_message(message: str) -> None:
                logger.debug("DC recv [%s]: %s", conn.pc_id, message[:80])
                asyncio.create_task(self._handle_data_message(conn, message))

    async def _read_audio_loop(self, conn: WebRTCConnection, track: Any) -> None:
        """Read audio frames from WebRTC inbound track → VoiceSession STT.

        Uses PyAV's AudioResampler for proper stateful streaming conversion:
        - Stereo → Mono (proper downmix, not just left channel)
        - 48kHz → 16kHz (anti-aliased, no per-chunk edge artifacts)
        - Any format → s16
        This matches pipecat's proven approach.
        """
        from av import AudioResampler

        resampler = AudioResampler(format="s16", layout="mono", rate=16000)

        try:
            while True:
                frame = await track.recv()
                if not conn.voice_session.is_streaming:
                    continue  # Drop until stream_start

                conn._audio_frame_count += 1

                # Resample: stereo 48kHz → mono 16kHz s16 (stateful, no edge artifacts)
                resampled_frames = resampler.resample(frame)

                for resampled in resampled_frames:
                    pcm_bytes = resampled.to_ndarray().tobytes()

                    if conn._audio_frame_count <= 3:
                        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
                        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                        logger.debug(
                            "Audio frame #%d: %d bytes, %d samples, rms=%.1f "
                            "(src: %s %s %dHz)",
                            conn._audio_frame_count,
                            len(pcm_bytes),
                            len(samples),
                            rms,
                            frame.format.name,
                            frame.layout.name,
                            frame.sample_rate,
                        )

                    if conn.vad is not None:
                        for event in conn.vad.process_pcm16(pcm_bytes):
                            await conn.voice_session.on_vad_event(event, conn.vad_mode)

                    await conn.voice_session.on_audio_in(pcm_bytes)

        except asyncio.CancelledError:
            pass  # Expected on connection close
        except Exception as e:
            logger.error(
                "Audio read loop error for %s: %s", conn.pc_id, e, exc_info=True
            )

    async def _handle_data_message(self, conn: WebRTCConnection, message: str) -> None:
        """Handle data channel messages from the client."""
        # Ignore non-JSON keepalives (e.g. bare "ping")
        if not message.startswith("{"):
            return

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type", "")
        try:
            if msg_type == "stream_start":
                if conn.voice_session.is_streaming:
                    # Already streaming — ignore duplicate start
                    pass
                elif conn.is_reconnect:
                    # Reconnect: resume existing session (no greeting, keep history)
                    await conn.voice_session.resume()
                else:
                    # First connect: full start with greeting
                    await conn.voice_session.start()

            elif msg_type == "stream_stop":
                # Explicit stop from client — fully tear down
                await conn.voice_session.stop()
                # Remove from persistent sessions so next connect starts fresh
                self._sessions.pop(conn.session_key, None)
                # If no sessions remain, clear the orchestrator keep_alive flag
                if not self._sessions and self.on_sessions_empty:
                    asyncio.create_task(self.on_sessions_empty())

            elif msg_type == "mute":
                conn.voice_session.mute()

            elif msg_type == "unmute":
                conn.voice_session.unmute()

            elif msg_type == "text":
                # Text input via data channel
                text = data.get("data", "")
                if text:
                    await conn.voice_session._trigger_response(text)

        except Exception as e:
            logger.error(
                "Data channel handler error (%s): %s", msg_type, e, exc_info=True
            )
            # Send error back to client so it's visible
            if conn.data_channel and conn.data_channel.readyState == "open":
                conn.data_channel.send(json.dumps({"type": "error", "data": str(e)}))

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

    CRITICAL: recv() must pace itself at real-time (~20ms per frame).
    aiortc's RTP sender calls recv() in a tight loop with no pacing —
    it relies on the track to block for the correct duration. Without
    pacing, silence frames burn through PTS at thousands-of-x realtime,
    and when real audio arrives the RTP timestamps are discontinuous,
    causing the browser to drop all audio.
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._sample_rate = 48000
        self._samples_per_frame = 960  # 20ms at 48kHz
        self._frame_duration = self._samples_per_frame / self._sample_rate  # 0.02s
        self._buffer = b""
        self._pts = 0
        self._start_time: float | None = None  # Wall-clock anchor for pacing
        self._audio_frames_served = 0  # Diagnostic counter
        self._silence_frames_served = 0

    def add_audio(self, audio_bytes: bytes) -> None:
        """Queue TTS audio bytes for playback."""
        self._queue.put_nowait(audio_bytes)

    def clear_audio(self) -> None:
        """Drop all queued and buffered outbound audio (barge-in)."""
        self._buffer = b""
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def recv(self) -> "AudioFrame":
        """Called by aiortc to get the next audio frame.

        Paces output at real-time by sleeping until the next frame is due.
        This keeps PTS aligned with wall-clock time so the browser's jitter
        buffer receives audio at the expected rate.
        """
        import time as _time

        # Anchor wall-clock on first call
        if self._start_time is None:
            self._start_time = _time.monotonic()

        # Calculate when this frame should be emitted
        frame_number = self._pts // self._samples_per_frame
        target_time = self._start_time + frame_number * self._frame_duration
        now = _time.monotonic()
        sleep_duration = target_time - now
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)

        frame_size = self._samples_per_frame * 2  # 16-bit = 2 bytes per sample

        # Fill buffer from queue
        had_audio = len(self._buffer) >= frame_size  # Already have leftover data
        while len(self._buffer) < frame_size:
            try:
                chunk = self._queue.get_nowait()
                self._buffer += chunk
                had_audio = True
            except asyncio.QueueEmpty:
                # No audio available — pad with silence
                self._buffer += b"\x00" * (frame_size - len(self._buffer))
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

        # Diagnostic logging
        if not had_audio:
            self._silence_frames_served += 1
        else:
            self._audio_frames_served += 1
            if self._audio_frames_served <= 5 or self._audio_frames_served % 100 == 0:
                logger.info(
                    "AudioOutputTrack: serving audio frame #%d (pts=%d, qsize=%d, buf=%d)",
                    self._audio_frames_served,
                    self._pts,
                    self._queue.qsize(),
                    len(self._buffer),
                )

        return frame


# ─── Audio Helpers ───────────────────────────────────────────────

# Use scipy for proper anti-aliased resampling (no aliasing artifacts).
# Falls back to numpy linear interp if scipy is somehow missing.
try:
    from math import gcd

    from scipy.signal import resample_poly as _scipy_resample_poly

    def _resample(pcm: "np.ndarray", src_rate: int, dst_rate: int) -> "np.ndarray":
        """Resample audio using polyphase anti-aliased filter (scipy)."""
        if src_rate == dst_rate:
            return pcm
        g = gcd(src_rate, dst_rate)
        up = dst_rate // g
        down = src_rate // g
        return _scipy_resample_poly(pcm, up, down).astype(pcm.dtype)

except ImportError:
    logger.warning(
        "scipy not installed — using linear interpolation for resampling (lower quality)"
    )

    def _resample(pcm: "np.ndarray", src_rate: int, dst_rate: int) -> "np.ndarray":  # type: ignore[misc]
        """Fallback linear resampling (no anti-aliasing)."""
        if src_rate == dst_rate:
            return pcm
        ratio = dst_rate / src_rate
        n_samples = int(len(pcm) * ratio)
        indices = np.linspace(0, len(pcm) - 1, n_samples)
        return np.interp(indices, np.arange(len(pcm)), pcm)
