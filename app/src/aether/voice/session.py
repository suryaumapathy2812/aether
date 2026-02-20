"""
VoiceSession — owns the full voice pipeline for one WebRTC connection.

Pipeline: mic audio → STT → transcript → AgentCore → TTS → speaker audio.

Turn detection state machine (mirrors LiveKit AudioRecognition):
─────────────────────────────────────────────────────────────────
  VAD START_OF_SPEECH
    → _speaking = True
    → cancel any pending EOU task (user is still talking)
    → barge-in if assistant is playing

  VAD END_OF_SPEECH
    → _speaking = False
    → _run_eou() — starts EOU task if transcript exists

  STT FINAL_TRANSCRIPT
    → accumulate into _transcript
    → if not _speaking (VAD already ended): _run_eou()
      (catch-up path: VAD fired before Deepgram delivered words)

  EOU task (_eou_task)
    → run turn detector once → get delay
    → sleep(delay)
    → if still valid (token match, not speaking): commit → response

  STT speech_started  → ignored for commit logic (noise signal only)
  STT utterance_end   → transcript accumulation only
  STT interim         → UI display only
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import unicodedata
from collections import deque
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from aether.core.config import config
from aether.core.frames import Frame, FrameType
from aether.core.metrics import metrics
from aether.providers.deepgram_stt import DeepgramSTTProvider
from aether.voice.turn_detection import TurnDecision, build_turn_detector

if TYPE_CHECKING:
    from aether.agent import AgentCore
    from aether.providers.base import TTSProvider

logger = logging.getLogger(__name__)

_STATE_IDLE_VAD_ONLY = "idle_vad_only"
_STATE_STT_CONNECTING = "stt_connecting"
_STATE_STT_STREAMING = "stt_streaming"
_STATE_THINKING = "thinking"
_STATE_TTS_PLAYING = "tts_playing"
_STATE_WATCHDOG_INTERVAL_S = 0.5
_STATE_STUCK_TIMEOUT_S = 8.0


class VoiceSession:
    """
    Owns the full voice pipeline for one WebRTC connection.

    One instance per peer connection, created by the WebRTC transport.
    """

    def __init__(
        self,
        agent: "AgentCore",
        tts_provider: "TTSProvider",
        session_id: str,
    ) -> None:
        self.agent = agent
        self.tts_provider = tts_provider
        self.session_id = session_id

        # Per-session STT
        self.stt = DeepgramSTTProvider()

        # ── Lifecycle flags ───────────────────────────────────────
        self.is_streaming = False
        self.is_responding = False
        self.is_muted = False
        self._stopped = False
        self._state = _STATE_IDLE_VAD_ONLY

        # ── Echo suppression ──────────────────────────────────────
        self._tts_playing = False
        self._tts_cooldown_until: float = 0.0
        self._assistant_speaking_until: float = 0.0

        # ── Turn detection state (the state machine) ──────────────
        self._speaking = False  # VAD says user is currently speaking
        self._transcript = ""  # accumulated finals for current turn
        self._last_final_text = ""  # dedup guard for STT finals
        self._eou_token: int = 0  # invalidates stale EOU tasks on new speech
        self._eou_task: asyncio.Task | None = None

        # ── Barge-in ──────────────────────────────────────────────
        self._last_barge_in_at: float = 0.0
        self._last_interim_at: float = 0.0
        self._last_interim_text: str = ""

        # ── Dialog history (for turn detector context) ────────────
        self._dialog_history: list[dict[str, str]] = []

        # ── Turn detector ─────────────────────────────────────────
        self._turn_detector = build_turn_detector(config.turn_detection)

        # ── Audio counters (logging) ──────────────────────────────
        self._audio_in_count = 0
        self._audio_in_dropped = 0

        # ── VAD-gated STT ────────────────────────────────────────────
        # STT connects only when speech is detected; ring buffer keeps
        # pre-speech audio so nothing is missed.
        self._stt_connected: bool = False
        _pre_roll_chunks = max(1, config.vad.stt_pre_roll_ms // 20)  # 20ms per chunk
        self._audio_ring_buffer: deque[bytes] = deque(maxlen=_pre_roll_chunks)
        self._stt_idle_disconnect_task: asyncio.Task | None = None

        # ── Notification queue ───────────────────────────────────────
        self._pending_notifications: list[str] = []

        # ── Background tasks ──────────────────────────────────────
        self._stt_event_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None
        self._turn_generation: int = 0
        self._watchdog_task: asyncio.Task | None = None
        self._barge_in_lock = asyncio.Lock()
        self._last_state_progress_at: float = time.monotonic()

        # ── Callbacks (set by WebRTC transport) ───────────────────
        self.on_audio_out: Callable[[bytes], Awaitable[None]] | None = None
        self.on_barge_in: Callable[[], Awaitable[None]] | None = None
        self.on_text_event: Callable[[dict], Awaitable[None]] | None = None
        # Called by the transport after resampling with the actual PCM duration.
        # More accurate than estimating from raw TTS bytes (which may be MP3/WAV).
        self.on_tts_duration: Callable[[float], None] | None = None

    # ─── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize STT and begin streaming according to VAD mode."""
        self._stopped = False
        logger.info(
            "[pipeline] %s start() — initializing STT (VAD-gated)", self.session_id
        )
        await self.stt.start()
        if config.vad.mode == "off":
            # No VAD events will arrive in this mode, so keep STT connected.
            logger.info("[pipeline] VAD off — connecting STT immediately")
            await self.stt.connect_stream()
            self._stt_connected = True
        else:
            # STT stream is NOT connected here — VAD speech_started will connect it.
            # This saves Deepgram costs during silence periods.
            self._stt_connected = False
        self.is_streaming = True
        if self._stt_connected:
            self._set_state(_STATE_STT_STREAMING, "session_start_vad_off")
        else:
            self._set_state(_STATE_IDLE_VAD_ONLY, "session_start")
        logger.info(
            "[pipeline] %s STT initialized, starting event loop", self.session_id
        )

        self._stt_event_task = asyncio.create_task(
            self._stt_event_loop(), name=f"stt-events-{self.session_id}"
        )
        self._ensure_watchdog_running()

        greeting = await self.agent.generate_greeting(
            session_id=self.session_id,
            is_resume=False,
        )
        if greeting:
            logger.info("[pipeline] %s greeting: %r", self.session_id, greeting[:60])
            await self._send_text_event("transcript", greeting, role="assistant")
            await self._synthesize_and_send(greeting)
            await self._send_text_event("stream_end", "")

        await self._send_text_event("status", "listening...")
        logger.info("[pipeline] %s started — ready for audio", self.session_id)

    async def stop(self) -> None:
        """Fully stop and clean up. Called on permanent session end."""
        self._stopped = True
        self.is_streaming = False
        self._set_state(_STATE_IDLE_VAD_ONLY, "session_stop")

        for task in (
            self._eou_task,
            self._stt_event_task,
            self._response_task,
            self._stt_idle_disconnect_task,
            self._watchdog_task,
        ):
            if task and not task.done():
                task.cancel()

        if self._stt_connected:
            await self.stt.disconnect_stream()
            self._stt_connected = False
        await self.stt.stop()
        await self.agent.cancel_session(self.session_id)

        logger.info("VoiceSession %s stopped", self.session_id)

    async def pause(self) -> None:
        """Pause on WebRTC disconnect — keep session state for fast reconnect.

        Stops audio/STT but preserves dialog history, transcript buffer,
        and agent session so the user can reconnect without losing context.
        """
        self.is_streaming = False

        # Cancel in-flight tasks but don't destroy session state
        for task in (
            self._eou_task,
            self._stt_event_task,
            self._response_task,
            self._stt_idle_disconnect_task,
            self._watchdog_task,
        ):
            if task and not task.done():
                task.cancel()
        self._eou_task = None
        self._stt_event_task = None
        self._stt_idle_disconnect_task = None
        self._watchdog_task = None

        # Disconnect STT stream if connected
        if self._stt_connected:
            try:
                await self.stt.disconnect_stream()
            except Exception:
                pass
            self._stt_connected = False

        # Clear audio callbacks — they point to the old peer connection
        self.on_audio_out = None
        self.on_barge_in = None
        self.on_text_event = None
        self.on_tts_duration = None

        # Reset speaking/responding state but keep _transcript and _dialog_history
        self._speaking = False
        self._tts_playing = False
        self._tts_cooldown_until = 0.0
        self._assistant_speaking_until = 0.0
        self.is_responding = False
        self._set_state(_STATE_IDLE_VAD_ONLY, "session_pause")

        logger.info("VoiceSession %s paused (reconnectable)", self.session_id)

    async def resume(self) -> None:
        """Resume after WebRTC reconnect — reuse existing session state.

        In VAD off mode, reconnect STT immediately.
        In VAD-gated modes, STT reconnect waits for speech_started.
        Callbacks must be re-wired by the transport before calling resume().
        """
        await self.stt.start()
        self._stopped = False
        logger.info("[pipeline] %s resume() — STT initialized", self.session_id)
        if self._stt_event_task is None or self._stt_event_task.done():
            logger.info(
                "[pipeline] %s resume() — restarting STT event loop", self.session_id
            )
            self._stt_event_task = asyncio.create_task(
                self._stt_event_loop(), name=f"stt-events-{self.session_id}"
            )

        if config.vad.mode == "off":
            logger.info("[pipeline] VAD off — reconnecting STT on resume")
            await self.stt.connect_stream()
            self._stt_connected = True
        else:
            # STT stays disconnected until VAD detects speech (VAD-gated)
            self._stt_connected = False
        self.is_streaming = True
        if self._stt_connected:
            self._set_state(_STATE_STT_STREAMING, "session_resume_vad_off")
        else:
            self._set_state(_STATE_IDLE_VAD_ONLY, "session_resume")
        if self._stt_connected and (
            self._stt_event_task is None or self._stt_event_task.done()
        ):
            # Defensive recovery: streaming STT must have a consumer loop.
            logger.info(
                "[pipeline] %s resume() — recovering STT event loop", self.session_id
            )
            self._stt_event_task = asyncio.create_task(
                self._stt_event_loop(), name=f"stt-events-{self.session_id}"
            )
        self._ensure_watchdog_running()

        # Generate greeting if appropriate (time-aware)
        greeting = await self.agent.generate_greeting(
            session_id=self.session_id,
            is_resume=True,
        )
        if greeting:
            logger.info(
                "[pipeline] %s resume greeting: %r", self.session_id, greeting[:60]
            )
            await self._send_text_event("transcript", greeting, role="assistant")
            await self._synthesize_and_send(greeting)
            await self._send_text_event("stream_end", "")

        await self._send_text_event("status", "listening...")
        logger.info(
            "VoiceSession %s resumed (%s)",
            self.session_id,
            "STT connected" if self._stt_connected else "STT will connect on speech",
        )

    # ─── Audio Input ─────────────────────────────────────────────

    async def on_audio_in(self, pcm_bytes: bytes) -> None:
        """Raw PCM from WebRTC → ring buffer → STT (when connected).

        Always appends to ring buffer (for pre-speech capture).
        Only forwards to STT when _stt_connected is True.
        Drops audio during TTS playback (echo suppression).
        """
        if not self.is_streaming or self.is_muted:
            return

        # Always buffer for pre-roll (VAD needs history when speech starts)
        self._audio_ring_buffer.append(pcm_bytes)

        if self._tts_playing or time.time() < self._tts_cooldown_until:
            self._audio_in_dropped += 1
            if self._audio_in_dropped <= 3 or self._audio_in_dropped % 200 == 0:
                logger.info(
                    "Echo suppression: dropped %d chunks (tts_playing=%s, cooldown=%.1fs)",
                    self._audio_in_dropped,
                    self._tts_playing,
                    max(0.0, self._tts_cooldown_until - time.time()),
                )
            return

        # Only forward to STT when the stream is explicitly active and writable.
        if not self._can_stream_stt_audio():
            return

        self._audio_in_count += 1
        if self._audio_in_count <= 3 or self._audio_in_count % 500 == 0:
            logger.info(
                "Audio to STT: chunk #%d (%d bytes)",
                self._audio_in_count,
                len(pcm_bytes),
            )
        await self.stt.send_audio(pcm_bytes)

    # ─── VAD Events ──────────────────────────────────────────────

    async def on_vad_event(self, event: dict[str, Any], mode: str) -> None:
        """
        VAD is the primary speech signal.

        START_OF_SPEECH → cancel pending EOU, mark speaking, barge-in
        END_OF_SPEECH   → mark not speaking, run EOU
        """
        action = event.get("action", "")
        probability = float(event.get("probability", 0.0))

        if action == "speech_started":
            self._speaking = True
            # Cancel any pending EOU — user is still talking
            self._cancel_eou()
            # Cancel pending STT disconnect — user is speaking again
            if (
                self._stt_idle_disconnect_task
                and not self._stt_idle_disconnect_task.done()
            ):
                self._stt_idle_disconnect_task.cancel()
                self._stt_idle_disconnect_task = None
            logger.debug("VAD speech_started (p=%.3f)", probability)

            # VAD-gated STT: connect on first speech detection.
            # If STT is already connected (common during barge-in), move
            # immediately back to streaming state so audio forwarding resumes.
            if self._stt_connected and self.stt.is_open:
                self._set_state(_STATE_STT_STREAMING, "vad_speech_started_resume")
            elif not self._stt_connected:
                await self._connect_stt_with_preroll()
            else:
                # Connected flag is stale (socket closed) — reconnect cleanly.
                self._stt_connected = False
                await self._connect_stt_with_preroll()

            if mode == "active":
                await self._handle_barge_in("vad", probability=probability)

        elif action == "speech_ended":
            self._speaking = False
            logger.info(
                "VAD speech_ended — transcript=%r",
                self._transcript.strip()[:60],
            )
            if not self.is_responding:
                self._run_eou()

        elif mode == "shadow":
            logger.debug("VAD %s (p=%.3f)", action, probability)

    # ─── VAD-Gated STT Connect/Disconnect ──────────────────────────

    async def _connect_stt_with_preroll(self) -> None:
        """Connect STT and flush the ring buffer (pre-speech audio)."""
        try:
            self._set_state(_STATE_STT_CONNECTING, "vad_speech_started")
            logger.info("[pipeline] Connecting STT (VAD triggered)")
            await self.stt.connect_stream()
            self._stt_connected = True
            self._set_state(_STATE_STT_STREAMING, "stt_connected")

            # Restart event loop if it was stopped
            if self._stt_event_task is None or self._stt_event_task.done():
                self._stt_event_task = asyncio.create_task(
                    self._stt_event_loop(), name=f"stt-events-{self.session_id}"
                )

            # Flush ring buffer — send buffered pre-speech audio
            buffered = list(self._audio_ring_buffer)
            if buffered:
                logger.info(
                    "[pipeline] Flushing %d pre-roll chunks to STT", len(buffered)
                )
                for chunk in buffered:
                    if self.stt.is_open:
                        await self.stt.send_audio(chunk)
        except Exception as e:
            logger.error("Failed to connect STT: %s", e, exc_info=True)
            self._stt_connected = False
            self._set_state(_STATE_IDLE_VAD_ONLY, "stt_connect_failed")

    def _schedule_stt_disconnect(self) -> None:
        """Schedule STT disconnect after idle timeout."""
        if self._stt_idle_disconnect_task and not self._stt_idle_disconnect_task.done():
            self._stt_idle_disconnect_task.cancel()
        self._stt_idle_disconnect_task = asyncio.create_task(
            self._disconnect_stt_idle(),
            name=f"stt-idle-disconnect-{self.session_id}",
        )

    async def _disconnect_stt_idle(self) -> None:
        """Disconnect STT after configurable silence period."""
        try:
            delay = config.vad.stt_idle_disconnect_s
            await asyncio.sleep(delay)
            if self._stt_connected and not self._speaking and not self.is_responding:
                logger.info("[pipeline] Disconnecting STT after %.1fs idle", delay)
                await self.stt.disconnect_stream()
                self._stt_connected = False
                self._set_state(_STATE_IDLE_VAD_ONLY, "stt_idle_disconnect")
        except asyncio.CancelledError:
            pass  # Speech started again, cancelled by on_vad_event

    # ─── STT Event Loop ──────────────────────────────────────────

    async def _stt_event_loop(self) -> None:
        """
        Consume STT frames. Three responsibilities:
          1. Accumulate finals into _transcript
          2. Catch-up EOU when final arrives after VAD already ended
          3. Forward interims to UI
        """
        try:
            async for frame in self.stt.stream_events():
                if not self.is_streaming:
                    break
                if self.is_muted:
                    continue

                if frame.type == FrameType.TEXT:
                    is_interim = frame.metadata.get("interim", False)

                    if is_interim:
                        if not self.is_responding:
                            text = str(frame.data)
                            self._last_interim_at = time.time()
                            self._last_interim_text = text
                            logger.info("STT interim: %r", text[:80])
                            await self._send_text_event(
                                "transcript", frame.data, interim=True
                            )

                    else:
                        # Final transcript
                        final = str(frame.data).strip()
                        if not final or final == self._last_final_text:
                            continue
                        self._last_final_text = final
                        self._transcript = (
                            f"{self._transcript} {final}".strip()
                            if self._transcript
                            else final
                        )
                        logger.info(
                            "[pipeline] STT final received: %r (speaking=%s)",
                            final[:80],
                            self._speaking,
                        )

                        if not self._speaking and not self.is_responding:
                            # Two cases both handled by _run_eou():
                            # 1. Catch-up: VAD ended before transcript arrived
                            #    (eou_task is None or done)
                            # 2. Transcript grew while EOU task was sleeping —
                            #    restart so the task commits the full transcript,
                            #    not the stale snapshot from when it started.
                            logger.info(
                                "STT final — restarting EOU with full transcript"
                            )
                            self._run_eou()

                elif frame.type == FrameType.CONTROL:
                    action = (
                        frame.data.get("action", "")
                        if isinstance(frame.data, dict)
                        else ""
                    )

                    if action == "utterance_end":
                        # Transcript accumulation only — no commit logic here.
                        # utterance_end often carries text already delivered
                        # via the final path — skip if already in _transcript
                        # to prevent duplicates.
                        transcript = (
                            frame.data.get("transcript", "")
                            if isinstance(frame.data, dict)
                            else ""
                        )
                        text = transcript.strip()
                        if (
                            text
                            and text != self._last_final_text
                            and text not in self._transcript
                        ):
                            self._last_final_text = text
                            self._transcript = (
                                f"{self._transcript} {text}".strip()
                                if self._transcript
                                else text
                            )
                        logger.debug("STT utterance_end: %r", transcript[:80])

                    elif action == "speech_started":
                        # Informational only. Deepgram fires this on background
                        # noise constantly — do not use for commit logic.
                        logger.debug("STT speech_started (noise guard — ignored)")
                        # Still check for barge-in during thinking phase
                        await self._handle_barge_in(
                            "stt", allow_thinking_interrupt=True
                        )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("STT event loop error: %s", e, exc_info=True)

    # ─── EOU State Machine ────────────────────────────────────────

    def _run_eou(self) -> None:
        """
        Start an EOU (end-of-utterance) task for the current transcript.

        Mirrors LiveKit _run_eou_detection:
        - Bail immediately if no transcript (VAD fired on silence/noise)
        - Cancel any existing EOU task first (fresh start)
        - Snapshot the token so stale tasks self-invalidate
        """
        if not self._transcript.strip():
            logger.debug("EOU skipped — transcript empty")
            return

        self._cancel_eou()
        self._eou_token += 1
        token = self._eou_token
        self._eou_task = asyncio.create_task(
            self._eou_task_fn(token),
            name=f"eou-{self.session_id}-{token}",
        )

    def _cancel_eou(self) -> None:
        """Invalidate and cancel any pending EOU task."""
        self._eou_token += 1  # stale tasks will see token mismatch and exit
        if self._eou_task and not self._eou_task.done():
            self._eou_task.cancel()
        self._eou_task = None

    async def _eou_task_fn(self, token: int) -> None:
        """
        EOU task: run turn detector once, sleep, then commit.

        IMPORTANT: do NOT snapshot _transcript at the start.
        The transcript grows while we sleep (Deepgram delivers finals
        asynchronously). Always read _transcript at commit time so we
        commit everything the user said, not just the first chunk.

        Token check at every await point — if VAD fires START_OF_SPEECH
        or a new final arrives while we sleep, _run_eou() increments the
        token and this task exits cleanly on the next check.
        """
        try:
            if token != self._eou_token:
                return

            # Snapshot for turn detector only — may be partial, that's ok.
            # The detector just needs enough context to decide the delay.
            text_for_detector = self._transcript.strip()
            if not text_for_detector:
                return

            # Run turn detector exactly once
            delay = await self._resolve_endpoint_delay(text_for_detector)

            if token != self._eou_token:
                return

            await asyncio.sleep(delay)

            if token != self._eou_token:
                return

            # Final guard: VAD must confirm silence
            if self._speaking:
                logger.info("EOU aborted after delay — VAD reports user speaking again")
                return

            # Read _transcript NOW — it may have grown during the sleep
            transcript = self._transcript.strip()
            self._transcript = ""
            self._last_final_text = ""
            if not transcript:
                return

            logger.info(
                "[pipeline] EOU committed — triggering LLM: %r", transcript[:80]
            )
            self._response_task = asyncio.create_task(
                self._trigger_response(transcript),
                name=f"voice-response-{self.session_id}",
            )

        except asyncio.CancelledError:
            pass  # Cancelled by _run_eou — new final arrived, restarting with more text

    # ─── Voice Response ──────────────────────────────────────────

    async def _trigger_response(self, text: str) -> None:
        """LLM → streaming TTS chunks → audio out."""
        generation = self._next_turn_generation()
        self.is_responding = True
        self._set_state(_STATE_THINKING, "llm_request_start")
        response_start = time.time()
        assistant_text = ""

        try:
            logger.info(
                "[pipeline] %s LLM request start: %r", self.session_id, text[:80]
            )
            await self._send_text_event("transcript", text, interim=False)
            await self._send_text_event("status", "thinking...")

            tts_buffer = ""

            async for event in self.agent.generate_reply_voice(text, self.session_id):
                if event.stream_type == "text_chunk":
                    chunk = event.payload.get("text", "")
                    await self._send_text_event("text_chunk", chunk)
                    assistant_text += chunk
                    tts_buffer += chunk
                    pieces, tts_buffer = _extract_tts_chunks(tts_buffer)
                    for piece in pieces:
                        await self._synthesize_and_send(piece, generation=generation)

                elif event.stream_type == "status":
                    status_text = event.payload.get("message", "")
                    if status_text:
                        await self._send_text_event("status", status_text)
                        await self._synthesize_and_send(
                            status_text, generation=generation
                        )

                elif event.stream_type == "tool_result":
                    await self._send_text_event(
                        "tool_result", json.dumps(event.payload)
                    )

            if tts_buffer.strip():
                await self._synthesize_and_send(
                    tts_buffer.strip(), generation=generation
                )

            await self._send_text_event("stream_end", "")

            self._append_dialog_turn("user", text)
            self._append_dialog_turn("assistant", assistant_text)

            metrics.observe("voice.response_ms", (time.time() - response_start) * 1000)

        except asyncio.CancelledError:
            logger.debug("Voice response cancelled (barge-in)")
        except Exception as e:
            logger.error("Voice response error: %s", e, exc_info=True)
            await self._send_text_event("error", str(e))
        finally:
            self.is_responding = False
            if self._stt_connected and self.stt.is_open:
                self._set_state(_STATE_STT_STREAMING, "response_complete")
            else:
                self._set_state(_STATE_IDLE_VAD_ONLY, "response_complete")
            if self.is_streaming:
                await self._send_text_event("status", "listening...")

            # Deliver any queued notifications
            if self._pending_notifications and self.is_streaming:
                for notif_text in self._pending_notifications:
                    logger.info(
                        "[pipeline] Delivering queued notification: %r", notif_text[:60]
                    )
                    await self._send_text_event(
                        "transcript", notif_text, role="assistant"
                    )
                    await self._synthesize_and_send(notif_text)
                    await self._send_text_event("stream_end", "")
                self._pending_notifications.clear()

            # Schedule STT disconnect after idle timeout
            if self._stt_connected and config.vad.mode in ("shadow", "active"):
                self._schedule_stt_disconnect()

            # If user spoke while we were responding, process it now
            if self._transcript.strip() and self.is_streaming:
                logger.info("Processing deferred utterance after response")
                self._run_eou()

    # ─── Turn Detector ────────────────────────────────────────────

    async def _resolve_endpoint_delay(self, text: str) -> float:
        """Ask turn detector for delay. Falls back to config defaults."""
        cfg = config.turn_detection
        default = cfg.max_endpointing_delay if cfg.mode == "active" else 0.5

        if not self._turn_detector:
            return default

        try:
            decision: TurnDecision = await self._turn_detector.predict_delay(
                self._dialog_history,
                text,
                language=None,
            )
            logger.info(
                "Turn detector: p=%.3f threshold=%.3f end=%s delay=%.2fs",
                decision.probability,
                decision.threshold,
                decision.likely_end_of_turn,
                decision.recommended_delay_s,
            )
            return decision.recommended_delay_s if cfg.mode == "active" else default
        except Exception as e:
            logger.warning(
                "Turn detector failed (%s) — using default %.1fs", e, default
            )
            return default

    def _append_dialog_turn(self, role: str, content: str) -> None:
        text = content.strip()
        if not text:
            return
        self._dialog_history.append({"role": role, "content": text})
        if len(self._dialog_history) > 12:
            self._dialog_history = self._dialog_history[-12:]

    # ─── Barge-in ────────────────────────────────────────────────

    async def _handle_barge_in(
        self,
        source: str,
        probability: float | None = None,
        allow_thinking_interrupt: bool = False,
    ) -> None:
        """Interrupt assistant playback when user starts speaking.

        Idempotent and race-safe: only one barge-in transaction runs at a time.
        """
        async with self._barge_in_lock:
            now = time.time()
            if not self._should_interrupt_for_barge_in(now, allow_thinking_interrupt):
                return

            self._last_barge_in_at = now
            if probability is not None:
                logger.info(
                    "Barge-in (%s, p=%.3f): interrupting assistant", source, probability
                )
            else:
                logger.info("Barge-in (%s): interrupting assistant", source)

            await self._reset_to_listening("barge_in", now)

    # ─── TTS ─────────────────────────────────────────────────────

    async def _synthesize_and_send(
        self, text: str, generation: int | None = None
    ) -> None:
        """Text → streaming TTS → audio out callback.

        Uses synthesize_stream() for chunk-by-chunk delivery, falling back
        to batch synthesis if the provider doesn't support streaming.

        Strips non-speakable characters (emoji, symbols) before synthesis so
        TTS providers don't waste a round-trip on unpronounceable glyphs.

        Echo suppression timing is set via on_tts_duration() called by the
        transport after resampling — that gives us the true PCM playback
        duration regardless of whether the provider returned MP3, WAV, or PCM.
        """
        clean = _clean_for_tts(text)
        if not clean or not self.on_audio_out:
            if text.strip() and not clean:
                logger.debug(
                    "TTS skipped — text contained only non-speakable chars: %r",
                    text[:60],
                )
            return
        if generation is not None and generation != self._turn_generation:
            logger.debug("TTS skipped for stale generation=%d", generation)
            return
        try:
            self._tts_playing = True
            self._set_state(_STATE_TTS_PLAYING, "tts_stream_start")
            logger.info("[pipeline] TTS stream start: %r", clean[:60])
            chunk_count = 0
            async for audio_chunk in self.tts_provider.synthesize_stream(clean):
                if generation is not None and generation != self._turn_generation:
                    logger.debug(
                        "TTS stream cancelled by generation change=%d", generation
                    )
                    break
                chunk_count += 1
                if chunk_count == 1:
                    logger.info(
                        "[pipeline] TTS first chunk: %d bytes for %r",
                        len(audio_chunk),
                        clean[:60],
                    )
                await self.on_audio_out(audio_chunk)
            logger.info("[pipeline] TTS stream done: %d chunks", chunk_count)
            # If the transport doesn't call on_tts_duration (e.g. tests),
            # fall back to a conservative 3-second guard.
            if not self.on_tts_duration:
                self._assistant_speaking_until = time.time() + 3.0
                self._tts_cooldown_until = time.time() + 3.3
        except Exception as e:
            logger.warning("TTS failed: %s", e, exc_info=True)
        finally:
            self._tts_playing = False
            if self.is_responding:
                self._set_state(_STATE_THINKING, "tts_stream_end")

    # ─── Helpers ─────────────────────────────────────────────────

    def _next_turn_generation(self) -> int:
        self._turn_generation += 1
        return self._turn_generation

    def _should_interrupt_for_barge_in(
        self, now: float, allow_thinking_interrupt: bool
    ) -> bool:
        if now - self._last_barge_in_at < 0.8:
            return False

        if now < self._assistant_speaking_until:
            return True

        if not allow_thinking_interrupt:
            return False
        if not (self._response_task and not self._response_task.done()):
            return False

        # Only interrupt thinking if there is fresh, substantive interim speech.
        has_fresh_interim = (
            now - self._last_interim_at < 1.2
            and len(self._last_interim_text.strip()) >= 6
        )
        return has_fresh_interim

    async def _reset_to_listening(self, reason: str, now: float | None = None) -> None:
        if self.on_barge_in:
            await self.on_barge_in()

        ts = now if now is not None else time.time()
        self._assistant_speaking_until = 0.0
        self._tts_playing = False
        self._tts_cooldown_until = ts + 0.35
        self._cancel_eou()

        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
        self._response_task = None

        self._turn_generation += 1
        self.is_responding = False
        if (
            self._stt_connected
            and self.stt.is_open
            and self.is_streaming
            and self._speaking
        ):
            self._set_state(_STATE_STT_STREAMING, reason)
        else:
            self._set_state(_STATE_IDLE_VAD_ONLY, reason)
        await self.agent.cancel_session(self.session_id)

        if self.is_streaming:
            await self._send_text_event("status", "listening...")

    def _ensure_watchdog_running(self) -> None:
        if self._watchdog_task and not self._watchdog_task.done():
            return
        self._watchdog_task = asyncio.create_task(
            self._state_watchdog_loop(),
            name=f"voice-watchdog-{self.session_id}",
        )

    async def _state_watchdog_loop(self) -> None:
        try:
            while self.is_streaming:
                await asyncio.sleep(_STATE_WATCHDOG_INTERVAL_S)
                if self._needs_watchdog_recovery():
                    logger.warning(
                        "[pipeline] state watchdog recovery (state=%s responding=%s)",
                        self._state,
                        self.is_responding,
                    )
                    await self._reset_to_listening("state_watchdog")
        except asyncio.CancelledError:
            pass

    def _needs_watchdog_recovery(self) -> bool:
        if self._speaking:
            return False

        elapsed = time.monotonic() - self._last_state_progress_at
        in_active_state = self._state in (_STATE_THINKING, _STATE_TTS_PLAYING)
        if in_active_state and elapsed > _STATE_STUCK_TIMEOUT_S:
            return True

        response_task_done = self._response_task is None or self._response_task.done()
        if self.is_responding and response_task_done and elapsed > 1.0:
            return True

        return False

    def _set_state(self, next_state: str, reason: str) -> None:
        self._last_state_progress_at = time.monotonic()
        if self._state == next_state:
            return
        prev = self._state
        self._state = next_state
        logger.info("[pipeline] state: %s -> %s (%s)", prev, next_state, reason)

    def _can_stream_stt_audio(self) -> bool:
        if self._state != _STATE_STT_STREAMING:
            return False
        if not self._stt_connected:
            return False
        if self.stt.is_open:
            return True
        logger.debug(
            "[pipeline] STT send skipped: state=%s stt_connected=%s stt_open=%s",
            self._state,
            self._stt_connected,
            self.stt.is_open,
        )
        return False

    async def _send_text_event(self, event_type: str, data: str, **kwargs: Any) -> None:
        if self.on_text_event:
            try:
                await self.on_text_event({"type": event_type, "data": data, **kwargs})
            except Exception as e:
                logger.debug("Failed to send text event %s: %s", event_type, e)

    async def deliver_notification(self, text: str) -> None:
        """Deliver a spoken notification through this voice session.

        If the session is currently responding, queue the notification
        for delivery after the response completes. If idle, synthesize
        and push audio immediately.
        """
        if not self.is_streaming:
            return

        if self.is_responding:
            # Queue for delivery after current response ends
            logger.info("[pipeline] Queuing notification (responding): %r", text[:60])
            self._pending_notifications.append(text)
        else:
            # Deliver immediately
            logger.info("[pipeline] Delivering notification now: %r", text[:60])
            await self._send_text_event("transcript", text, role="assistant")
            await self._synthesize_and_send(text)
            await self._send_text_event("stream_end", "")

    def mute(self) -> None:
        self.is_muted = True

    def unmute(self) -> None:
        self.is_muted = False


_TTS_CHUNK_MIN_CHARS = 60
_TTS_CHUNK_MAX_CHARS = 140
_TTS_PAUSE_CHARS = ".,;:!?\n"


def _extract_tts_chunks(buffer: str) -> tuple[list[str], str]:
    """Extract speakable chunks from a streamed LLM text buffer.

    This intentionally avoids sentence-boundary trimming. It emits on natural
    pause punctuation when available, otherwise falls back to a max size cut.
    """
    pieces: list[str] = []
    remaining = buffer

    while len(remaining) >= _TTS_CHUNK_MIN_CHARS:
        pause_cut = max(remaining.rfind(ch) for ch in _TTS_PAUSE_CHARS)
        if pause_cut >= _TTS_CHUNK_MIN_CHARS:
            cut = pause_cut + 1
        elif len(remaining) >= _TTS_CHUNK_MAX_CHARS:
            cut = remaining.rfind(" ", 0, _TTS_CHUNK_MAX_CHARS)
            if cut == -1:
                cut = _TTS_CHUNK_MAX_CHARS
        else:
            break

        piece = remaining[:cut].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[cut:].lstrip()

    return pieces, remaining


# Unicode categories that are non-speakable (emoji, symbols, surrogates, etc.)
_NON_SPEAKABLE_CATEGORIES = frozenset(
    {
        "So",  # Other symbol (emoji, misc symbols)
        "Sm",  # Math symbol
        "Sk",  # Modifier symbol
        "Cs",  # Surrogate
        "Co",  # Private use
        "Cn",  # Unassigned
    }
)

# Emoji ranges not always caught by category (e.g. ZWJ sequences, variation selectors)
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # misc symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f700-\U0001f77f"  # alchemical
    "\U0001f780-\U0001f7ff"  # geometric shapes extended
    "\U0001f800-\U0001f8ff"  # supplemental arrows-c
    "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
    "\U0001fa00-\U0001fa6f"  # chess symbols
    "\U0001fa70-\U0001faff"  # symbols & pictographs extended-a
    "\U00002702-\U000027b0"  # dingbats
    "\U000024c2-\U0001f251"  # enclosed chars
    "\ufe0f"  # variation selector-16 (emoji presentation)
    "\u200d"  # zero-width joiner
    "]+",
    flags=re.UNICODE,
)


def _clean_for_tts(text: str) -> str:
    """Strip emoji and non-speakable Unicode characters before TTS synthesis.

    Removes:
    - Emoji (via regex covering all major Unicode emoji blocks)
    - Characters in non-speakable Unicode categories (So, Sm, Sk, Cs, Co, Cn)

    Preserves all normal Latin, CJK, Devanagari, punctuation, digits, etc.
    Collapses any resulting double-spaces and strips leading/trailing whitespace.
    """
    # Remove emoji sequences first (handles ZWJ sequences, variation selectors)
    text = _EMOJI_RE.sub("", text)
    # Remove remaining non-speakable category chars
    text = "".join(
        ch for ch in text if unicodedata.category(ch) not in _NON_SPEAKABLE_CATEGORIES
    )
    # Collapse multiple spaces left by removed chars
    text = re.sub(r"  +", " ", text).strip()
    return text
