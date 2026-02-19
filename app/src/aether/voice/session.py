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

        # ── Background tasks ──────────────────────────────────────
        self._stt_event_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None

        # ── Callbacks (set by WebRTC transport) ───────────────────
        self.on_audio_out: Callable[[bytes], Awaitable[None]] | None = None
        self.on_barge_in: Callable[[], Awaitable[None]] | None = None
        self.on_text_event: Callable[[dict], Awaitable[None]] | None = None

    # ─── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Connect STT, start event loop, play greeting."""
        await self.stt.start()
        await self.stt.connect_stream()
        self.is_streaming = True

        self._stt_event_task = asyncio.create_task(
            self._stt_event_loop(), name=f"stt-events-{self.session_id}"
        )

        greeting = await self.agent.generate_greeting()
        if greeting:
            await self._send_text_event("transcript", greeting, role="assistant")
            await self._synthesize_and_send(greeting)
            await self._send_text_event("stream_end", "")

        await self._send_text_event("status", "listening...")
        logger.info("VoiceSession %s started", self.session_id)

    async def stop(self) -> None:
        """Disconnect and clean up all tasks."""
        self.is_streaming = False

        for task in (self._eou_task, self._stt_event_task, self._response_task):
            if task and not task.done():
                task.cancel()

        await self.stt.disconnect_stream()
        await self.stt.stop()
        await self.agent.cancel_session(self.session_id)

        logger.info("VoiceSession %s stopped", self.session_id)

    # ─── Audio Input ─────────────────────────────────────────────

    async def on_audio_in(self, pcm_bytes: bytes) -> None:
        """Raw PCM from WebRTC → STT. Drops audio during TTS playback."""
        if not self.is_streaming or self.is_muted:
            return

        if self._tts_playing or time.time() < self._tts_cooldown_until:
            self._audio_in_dropped += 1
            if self._audio_in_dropped <= 3 or self._audio_in_dropped % 200 == 0:
                logger.info(
                    "Echo suppression: dropped %d chunks (tts_playing=%s, cooldown=%.1fs)",
                    self._audio_in_dropped,
                    self._tts_playing,
                    max(0.0, self._tts_cooldown_until - time.time()),
                )
            # Send silence to keep Deepgram websocket alive
            await self.stt.send_audio(b"\x00" * len(pcm_bytes))
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
            logger.debug("VAD speech_started (p=%.3f)", probability)

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
                            "STT final: %r (speaking=%s)", final[:80], self._speaking
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

            logger.info("EOU committed: %r", transcript[:80])
            self._response_task = asyncio.create_task(
                self._trigger_response(transcript),
                name=f"voice-response-{self.session_id}",
            )

        except asyncio.CancelledError:
            pass  # Cancelled by _run_eou — new final arrived, restarting with more text

    # ─── Voice Response ──────────────────────────────────────────

    async def _trigger_response(self, text: str) -> None:
        """LLM → sentence-chunked TTS → audio out."""
        self.is_responding = True
        response_start = time.time()
        assistant_text = ""

        try:
            await self._send_text_event("transcript", text, interim=False)
            await self._send_text_event("status", "thinking...")

            sentence_buffer = ""
            async for event in self.agent.generate_reply_voice(text, self.session_id):
                if event.stream_type == "text_chunk":
                    chunk = event.payload.get("text", "")
                    await self._send_text_event("text_chunk", chunk)
                    assistant_text += chunk

                    sentence_buffer += chunk
                    sentences = _split_sentences(sentence_buffer)
                    for sentence in sentences[:-1]:
                        if sentence.strip():
                            await self._synthesize_and_send(sentence.strip())
                    sentence_buffer = sentences[-1] if sentences else ""

                elif event.stream_type == "status":
                    status_text = event.payload.get("message", "")
                    if status_text:
                        await self._send_text_event("status", status_text)
                        await self._synthesize_and_send(status_text)

                elif event.stream_type == "tool_result":
                    await self._send_text_event(
                        "tool_result", json.dumps(event.payload)
                    )

            if sentence_buffer.strip():
                await self._synthesize_and_send(sentence_buffer.strip())

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
            if self.is_streaming:
                await self._send_text_event("status", "listening...")

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
        """Interrupt assistant playback when user starts speaking."""
        now = time.time()
        if now - self._last_barge_in_at < 0.8:
            return

        is_assistant_speaking = now < self._assistant_speaking_until
        if not is_assistant_speaking:
            if not allow_thinking_interrupt:
                return
            if not (self._response_task and not self._response_task.done()):
                return
            # Only interrupt thinking if there's a real interim transcript
            has_fresh_interim = (
                now - self._last_interim_at < 1.2
                and len(self._last_interim_text.strip()) >= 6
            )
            if not has_fresh_interim:
                return

        self._last_barge_in_at = now

        if probability is not None:
            logger.info(
                "Barge-in (%s, p=%.3f): interrupting assistant", source, probability
            )
        else:
            logger.info("Barge-in (%s): interrupting assistant", source)

        if self.on_barge_in:
            await self.on_barge_in()

        self._assistant_speaking_until = 0.0
        self._tts_playing = False
        self._tts_cooldown_until = now + 0.35

        if self._response_task and not self._response_task.done():
            self._response_task.cancel()

        self.is_responding = False
        await self.agent.cancel_session(self.session_id)

        if self.is_streaming:
            await self._send_text_event("status", "listening...")

    # ─── TTS ─────────────────────────────────────────────────────

    async def _synthesize_and_send(self, text: str) -> None:
        """Text → TTS → audio out callback."""
        if not text.strip() or not self.on_audio_out:
            return
        try:
            self._tts_playing = True
            audio_bytes = await self.tts_provider.synthesize(text)
            logger.info(
                "TTS: %d bytes (%.1fs) for %r",
                len(audio_bytes),
                len(audio_bytes) / 48000.0,
                text[:60],
            )
            await self.on_audio_out(audio_bytes)
            logger.info("TTS audio queued")
            playback_secs = len(audio_bytes) / 48000.0
            self._assistant_speaking_until = time.time() + playback_secs
            self._tts_cooldown_until = time.time() + playback_secs + 0.3
        except Exception as e:
            logger.warning("TTS failed: %s", e, exc_info=True)
        finally:
            self._tts_playing = False

    # ─── Helpers ─────────────────────────────────────────────────

    async def _send_text_event(self, event_type: str, data: str, **kwargs: Any) -> None:
        if self.on_text_event:
            try:
                await self.on_text_event({"type": event_type, "data": data, **kwargs})
            except Exception as e:
                logger.debug("Failed to send text event %s: %s", event_type, e)

    def mute(self) -> None:
        self.is_muted = True

    def unmute(self) -> None:
        self.is_muted = False


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries for low-latency TTS chunking."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts if parts else [text]
