"""
Simplified WebRTC transport — signaling + audio routing.

Creates a VoiceSession per peer connection. No CoreMsg, no TransportManager.
Direct: audio frames → VoiceSession → realtime model → audio out.

Requires aiortc (optional dependency). If not installed, the transport
is not available and main.py skips it gracefully.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from aether.core.config import config
from aether.voice.io import (
    AudioFrame as IOAudioFrame,
    AudioInput,
    AudioInputEvent,
    AudioOutput,
    NullTextOutput,
    TextOutput,
)
from aether.voice.realtime import RealtimeModel
from aether.voice.session import VoiceSession
from aether.voice.vad import VADSettings, build_vad

if TYPE_CHECKING:
    from aether.agent import AgentCore
    from aether.providers.base import TTSProvider
    from aether.session.ledger import TaskLedger
    from aether.session.store import SessionStore

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
    from aiortc.mediastreams import MediaStreamError
    from av import AudioFrame

    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    np = None  # type: ignore[assignment]

    class MediaStreamTrack:  # type: ignore[no-redef]
        """Fallback base to allow module import without aiortc."""

    class MediaStreamError(Exception):
        """Fallback media stream error when aiortc is unavailable."""


@dataclass
class WebRTCConnection:
    """State for one WebRTC peer connection."""

    pc_id: str
    pc: Any  # RTCPeerConnection
    voice_session: VoiceSession
    audio_input: Any
    audio_output: Any
    text_output: Any
    audio_out_track: Any  # AudioOutputTrack
    data_channel: Any | None = None
    vad: Any | None = None
    vad_mode: str = "off"
    _audio_frame_count: int = field(default=0, repr=False)
    is_reconnect: bool = False  # True if reusing an existing VoiceSession
    session_key: str = ""  # Key into WebRTCVoiceTransport._sessions
    device_id: str = ""
    _disconnect_grace_task: Any | None = field(default=None, repr=False)
    last_activity_at: float = field(default_factory=lambda: __import__("time").time())
    offer_epoch: int = 0


class WebRTCAudioInput(AudioInput):
    """Queue-backed AudioInput for WebRTC inbound PCM frames."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[IOAudioFrame] = asyncio.Queue(maxsize=200)
        self._closed = False

    async def push_pcm(
        self,
        pcm_bytes: bytes,
        *,
        event: AudioInputEvent | None = None,
    ) -> None:
        if self._closed:
            return
        frame = IOAudioFrame(
            data=pcm_bytes,
            sample_rate=16000,
            samples_per_channel=len(pcm_bytes) // 2,
            event=event,
        )
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            return

    async def __aiter__(self):
        while not self._closed:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield frame
            except asyncio.TimeoutError:
                continue

    async def close(self) -> None:
        self._closed = True


class WebRTCAudioOutput(AudioOutput):
    """AudioOutput backed by AudioOutputTrack with 24kHz->48kHz resampling."""

    def __init__(self, track: Any) -> None:
        from av import AudioResampler

        self._track = track
        self._resampler = AudioResampler(format="s16", layout="mono", rate=48000)

    async def push_frame(self, frame: IOAudioFrame) -> None:
        sample_count = len(frame.data) // 2
        src_frame = AudioFrame(format="s16", layout="mono", samples=sample_count)
        src_frame.sample_rate = frame.sample_rate
        src_frame.planes[0].update(frame.data)
        for out in self._resampler.resample(src_frame):
            self._track.add_audio(out.to_ndarray().tobytes())

    async def clear(self) -> None:
        self._track.clear_audio()

    async def close(self) -> None:
        return None


class WebRTCTextOutput(TextOutput):
    """TextOutput for sending session events over the data channel."""

    def __init__(self, data_channel: Any) -> None:
        self._dc = data_channel

    async def push_text(self, text: str, *, final: bool = False) -> None:
        if self._dc is None or self._dc.readyState != "open":
            return
        if text:
            self._dc.send(json.dumps({"type": "text_chunk", "data": text}))
        if final:
            self._dc.send(json.dumps({"type": "stream_end", "data": ""}))

    async def push_state(self, state: str) -> None:
        if self._dc is None or self._dc.readyState != "open":
            return
        self._dc.send(json.dumps({"type": "status", "data": state}))


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
        realtime_model: RealtimeModel | None = None,
        on_function_call: Callable[[str, str, str], Awaitable[str]] | None = None,
        instructions: str = "",
        tools: list[dict[str, Any]] | None = None,
        session_store: "SessionStore | None" = None,
        task_ledger: "TaskLedger | None" = None,
    ) -> None:
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc is required for WebRTC transport")

        self.agent = agent
        self.tts_provider = tts_provider
        self.realtime_model = realtime_model
        self.on_function_call = on_function_call
        self.instructions = instructions
        self.tools = tools
        self.session_store = session_store
        self.task_ledger = task_ledger
        self.ice_servers = ice_servers or [{"urls": "stun:stun.l.google.com:19302"}]
        self._connections: dict[str, WebRTCConnection] = {}
        # Persistent sessions keyed by user_id — survive WebRTC disconnects
        self._sessions: dict[str, VoiceSession] = {}
        # Monotonic offer epoch per logical session key.
        # Used to make handoff ordering explicit and deterministic.
        self._offer_epoch: dict[str, int] = {}
        # Optional callback: called with False when all persistent sessions are gone.
        # Wire this up in main.py to clear the orchestrator keep_alive flag.
        self.on_sessions_empty: Callable[[], Any] | None = None

    async def handle_offer(
        self,
        sdp: str,
        sdp_type: str = "offer",
        user_id: str = "",
        device_id: str = "",
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
        owner_device_id = device_id or f"anonymous:{pc_id}"
        offer_epoch = self._next_offer_epoch(session_key)

        # Canonical session arbitration: keep only one active peer connection
        # per logical session key. New offer wins (last-write-wins).
        handoff_from = await self._evict_connections_for_session(
            session_key,
            winner_device_id=owner_device_id,
            winner_pc_id=pc_id,
            winner_epoch=offer_epoch,
        )

        is_reconnect = session_key in self._sessions
        if is_reconnect:
            voice_session = self._sessions[session_key]
            logger.info(
                "WebRTC reconnect for %s — reusing existing VoiceSession", session_key
            )
        else:
            if self.realtime_model is None:
                raise RuntimeError("Realtime model is not configured for WebRTC")
            voice_session = VoiceSession(
                session_id=f"webrtc-{pc_id}",
                realtime_model=self.realtime_model,
                on_function_call=self.on_function_call,
                instructions=self.instructions,
                tools=self.tools,
                session_store=self.session_store,
                task_ledger=self.task_ledger,
            )
            self._sessions[session_key] = voice_session

        # Create outbound audio track
        audio_out = AudioOutputTrack()
        pc.addTrack(audio_out)
        audio_input = WebRTCAudioInput()
        audio_output = WebRTCAudioOutput(audio_out)
        text_output: TextOutput = NullTextOutput()
        voice_session.set_io(audio_input, audio_output, text_output)

        # Store connection
        conn = WebRTCConnection(
            pc_id=pc_id,
            pc=pc,
            voice_session=voice_session,
            audio_input=audio_input,
            audio_output=audio_output,
            text_output=text_output,
            audio_out_track=audio_out,
            vad_mode=config.vad.mode,
            is_reconnect=is_reconnect,
            session_key=session_key,
            device_id=owner_device_id,
            offer_epoch=offer_epoch,
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
            "active_device_id": owner_device_id,
            "handoff_from_device_id": handoff_from,
        }

    def _next_offer_epoch(self, session_key: str) -> int:
        next_epoch = self._offer_epoch.get(session_key, 0) + 1
        self._offer_epoch[session_key] = next_epoch
        return next_epoch

    async def _evict_connections_for_session(
        self,
        session_key: str,
        *,
        winner_device_id: str,
        winner_pc_id: str,
        winner_epoch: int,
    ) -> str | None:
        stale_connections = [
            (pc_id, conn)
            for pc_id, conn in self._connections.items()
            if conn.session_key == session_key
        ]
        stale_pc_ids = [pc_id for pc_id, _ in stale_connections]
        if not stale_pc_ids:
            return None

        previous_owner = stale_connections[0][1].device_id or None

        for stale_pc_id in stale_pc_ids:
            stale = self._connections.pop(stale_pc_id, None)
            if stale is None:
                continue
            logger.info(
                "Session handoff: session=%s old_pc=%s old_device=%s new_pc=%s new_device=%s epoch=%s",
                session_key,
                stale_pc_id,
                stale.device_id or "unknown",
                winner_pc_id,
                winner_device_id,
                winner_epoch,
            )
            try:
                if (
                    stale._disconnect_grace_task
                    and not stale._disconnect_grace_task.done()
                ):
                    stale._disconnect_grace_task.cancel()
                await stale.audio_input.close()
                await stale.audio_output.close()
                await stale.pc.close()
            except Exception:
                logger.debug(
                    "Failed evicting stale connection %s", stale_pc_id, exc_info=True
                )
        return previous_owner

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

            if state == "connected":
                # Cancel any pending grace timer
                if (
                    conn._disconnect_grace_task
                    and not conn._disconnect_grace_task.done()
                ):
                    conn._disconnect_grace_task.cancel()
                    conn._disconnect_grace_task = None
                conn.last_activity_at = time.time()

            elif state == "disconnected":
                # Start grace timer — connection may recover
                grace_s = config.webrtc.disconnect_grace_seconds
                logger.info(
                    "WebRTC %s disconnected — starting %ds grace timer",
                    conn.pc_id,
                    grace_s,
                )
                conn._disconnect_grace_task = asyncio.create_task(
                    self._disconnect_grace(conn, grace_s)
                )

            elif state in ("failed", "closed"):
                # Immediate pause — no grace period
                if (
                    conn._disconnect_grace_task
                    and not conn._disconnect_grace_task.done()
                ):
                    conn._disconnect_grace_task.cancel()
                await conn.voice_session.pause()
                self._connections.pop(conn.pc_id, None)
                logger.info(
                    "WebRTC %s %s — session paused, ready for reconnect",
                    conn.pc_id,
                    state,
                )

        @pc.on("track")
        async def on_track(track: Any) -> None:
            if track.kind == "audio":

                @track.on("ended")
                async def on_track_ended() -> None:
                    logger.info("WebRTC %s inbound audio track ended", conn.pc_id)

                asyncio.create_task(
                    self._read_audio_loop(conn, track),
                    name=f"audio-in-{conn.pc_id}",
                )

        @pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            logger.info("Data channel opened: %s", conn.pc_id)
            conn.data_channel = channel
            conn.text_output = WebRTCTextOutput(channel)
            conn.voice_session.set_io(
                conn.audio_input, conn.audio_output, conn.text_output
            )

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
                            action = str(event.get("action", ""))
                            if action == "speech_started":
                                await conn.audio_input.push_pcm(
                                    b"", event=AudioInputEvent.START_OF_SPEECH
                                )
                            elif action == "speech_ended":
                                await conn.audio_input.push_pcm(
                                    b"", event=AudioInputEvent.END_OF_SPEECH
                                )

                    await conn.audio_input.push_pcm(pcm_bytes)

        except asyncio.CancelledError:
            pass  # Expected on connection close
        except MediaStreamError:
            # Expected when the remote peer closes/stops the inbound audio track.
            logger.info(
                "Audio read loop closed for %s (MediaStreamError, pc_state=%s)",
                conn.pc_id,
                conn.pc.connectionState,
            )
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
                    # only when it wasn't fully stopped before.
                    if getattr(conn.voice_session, "_stopped", False):
                        logger.info(
                            "stream_start on reconnect (%s): session marked stopped, starting fresh",
                            conn.pc_id,
                        )
                        await conn.voice_session.start()
                    else:
                        logger.info(
                            "stream_start on reconnect (%s): resuming session",
                            conn.pc_id,
                        )
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
                    await conn.voice_session.inject_text(text)

        except Exception as e:
            logger.error(
                "Data channel handler error (%s): %s", msg_type, e, exc_info=True
            )
            # Send error back to client so it's visible
            if conn.data_channel and conn.data_channel.readyState == "open":
                conn.data_channel.send(json.dumps({"type": "error", "data": str(e)}))

    def get_active_sessions(self) -> list[VoiceSession]:
        """Return all active (streaming) voice sessions."""
        return [session for session in self._sessions.values() if session.is_streaming]

    async def _disconnect_grace(self, conn: WebRTCConnection, grace_s: int) -> None:
        """Wait for connection recovery, then pause if still disconnected."""
        try:
            await asyncio.sleep(grace_s)
            # If still disconnected after grace period, pause the session
            state = conn.pc.connectionState
            if state in ("disconnected", "failed", "closed"):
                logger.info(
                    "WebRTC %s grace period expired (state=%s) — pausing session",
                    conn.pc_id,
                    state,
                )
                await conn.voice_session.pause()
                self._connections.pop(conn.pc_id, None)
        except asyncio.CancelledError:
            pass  # Connection recovered or session was cleaned up

    async def start_session_ttl_sweep(self) -> None:
        """Start periodic sweep task that removes idle sessions."""
        self._sweep_task = asyncio.create_task(
            self._session_ttl_sweep(),
            name="webrtc-session-ttl-sweep",
        )

    async def _session_ttl_sweep(self) -> None:
        """Periodically remove sessions that have been idle too long."""
        ttl = config.webrtc.session_ttl_seconds
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                now = time.time()
                expired_keys = []

                for key, session in self._sessions.items():
                    # Check if session has an active connection
                    has_active_conn = any(
                        c.session_key == key for c in self._connections.values()
                    )
                    if not has_active_conn and not session.is_streaming:
                        # Check how long since session was paused
                        # Use a simple heuristic: if not streaming and no
                        # active connection for TTL seconds, clean up
                        expired_keys.append(key)

                for key in expired_keys:
                    session = self._sessions.pop(key, None)
                    if session:
                        logger.info("Session TTL expired for %s — cleaning up", key)
                        await session.stop()

                if expired_keys and not self._sessions and self.on_sessions_empty:
                    asyncio.create_task(self.on_sessions_empty())

        except asyncio.CancelledError:
            pass

    async def close_all(self) -> None:
        """Close all connections (shutdown)."""
        # Cancel sweep task
        if hasattr(self, "_sweep_task") and self._sweep_task:
            self._sweep_task.cancel()

        for conn in list(self._connections.values()):
            try:
                await conn.voice_session.stop()
                await conn.pc.close()
            except Exception:
                pass
        self._connections.clear()
        self._sessions.clear()


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
