"""
VoiceSession — owns the full voice pipeline for one WebRTC connection.

Each session creates its own STT instance (no shared singleton).
Pipeline: mic audio → STT → transcript → AgentCore → TTS → speaker audio.

Key design:
- Per-session STT fixes the shared singleton bug
- Sentence-level TTS chunking for low latency
- Debounced utterance triggering (wait for silence before responding)
- Barge-in: new speech cancels in-progress response
- Mute/unmute support
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

    Each session creates its own STT instance (no shared singleton).
    The WebRTC transport creates one VoiceSession per peer connection.
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

        # Per-session STT (fixes shared singleton bug)
        self.stt = DeepgramSTTProvider()

        # State
        self.is_streaming = False
        self.is_responding = False
        self.is_muted = False
        self._tts_playing = False  # Echo suppression: mute STT input during TTS
        self._tts_cooldown_until: float = 0.0  # Timestamp: ignore mic until echo fades
        self._assistant_speaking_until: float = 0.0
        self._last_barge_in_at: float = 0.0
        self._vad_speaking = False
        self._last_interim_at: float = 0.0
        self._last_interim_text: str = ""
        self._last_final_text: str = ""
        self._last_user_activity_at: float = time.time()
        self._turn_commit_token: int = 0
        self._audio_in_count = 0
        self._audio_in_dropped = 0
        self.accumulated_transcript = ""
        self._dialog_history: list[dict[str, str]] = []
        self._turn_detector = build_turn_detector(config.turn_detection)
        self._debounce_task: asyncio.Task | None = None
        self._stt_event_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None

        # Callbacks (set by WebRTC transport)
        self.on_audio_out: Callable[[bytes], Awaitable[None]] | None = None
        self.on_barge_in: Callable[[], Awaitable[None]] | None = None
        self.on_text_event: Callable[[dict], Awaitable[None]] | None = None

    # ─── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Called on stream_start. Connect STT, start listening."""
        # Start per-session STT — must connect before enabling audio flow
        await self.stt.start()
        await self.stt.connect_stream()

        # Now enable audio flow (audio frames are dropped until this is True)
        self.is_streaming = True

        self._stt_event_task = asyncio.create_task(
            self._stt_event_loop(), name=f"stt-events-{self.session_id}"
        )

        # Greeting (first connection only)
        greeting = await self.agent.generate_greeting()
        if greeting:
            await self._send_text_event("transcript", greeting, role="assistant")
            await self._synthesize_and_send(greeting)
            await self._send_text_event("stream_end", "")

        await self._send_text_event("status", "listening...")
        logger.info("VoiceSession %s started", self.session_id)

    async def stop(self) -> None:
        """Called on disconnect. Cleanup everything."""
        self.is_streaming = False

        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        if self._stt_event_task and not self._stt_event_task.done():
            self._stt_event_task.cancel()
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()

        await self.stt.disconnect_stream()
        await self.stt.stop()

        # Cancel any pending jobs for this session
        await self.agent.cancel_session(self.session_id)

        logger.info("VoiceSession %s stopped", self.session_id)

    # ─── Audio Input ─────────────────────────────────────────────

    async def on_audio_in(self, pcm_bytes: bytes) -> None:
        """Raw PCM from WebRTC audio track → STT."""
        if not self.is_streaming or self.is_muted:
            return
        # Echo suppression: drop mic audio while TTS is playing (+ 300ms cooldown)
        if self._tts_playing or time.time() < self._tts_cooldown_until:
            self._audio_in_dropped += 1
            if self._audio_in_dropped <= 3 or self._audio_in_dropped % 200 == 0:
                logger.info(
                    "Echo suppression: dropped %d audio chunks (tts_playing=%s, cooldown_remaining=%.1fs)",
                    self._audio_in_dropped,
                    self._tts_playing,
                    max(0, self._tts_cooldown_until - time.time()),
                )
            # Keep STT websocket alive during suppression windows.
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

    async def on_vad_event(self, event: dict[str, Any], mode: str) -> None:
        """Handle optional VAD events from WebRTC layer.

        - shadow: log only
        - active: allow barge-in cancellation when assistant is actively speaking
        """
        action = event.get("action", "")
        probability = float(event.get("probability", 0.0))

        if action == "speech_started":
            self._vad_speaking = True
            self._last_user_activity_at = time.time()
        elif action == "speech_ended":
            self._vad_speaking = False
            logger.info(
                "VAD speech_ended — transcript=%r responding=%s",
                self.accumulated_transcript.strip()[:60],
                self.is_responding,
            )
            # VAD speech_ended is the authoritative commit trigger.
            # Schedule regardless of whether utterance_end was seen —
            # _debounce_and_trigger will bail early if transcript is empty.
            if not self.is_responding:
                self._schedule_turn_commit()

        if mode == "shadow":
            logger.debug("VAD %s (p=%.3f)", action, probability)
            return

        if mode != "active":
            return

        if action != "speech_started":
            return

        await self._handle_barge_in("vad", probability=probability)

    # ─── STT Event Loop ──────────────────────────────────────────

    async def _stt_event_loop(self) -> None:
        """Listen to STT events, trigger response on utterance end."""
        try:
            async for frame in self.stt.stream_events():
                if not self.is_streaming:
                    break
                if self.is_muted:
                    continue

                if frame.type == FrameType.TEXT:
                    # Interim transcript
                    is_interim = frame.metadata.get("interim", False)
                    if is_interim and not self.is_responding:
                        logger.info("STT interim: %r", str(frame.data)[:80])
                        self._last_user_activity_at = time.time()
                        self._last_interim_at = time.time()
                        self._last_interim_text = str(frame.data)
                        await self._send_text_event(
                            "transcript", frame.data, interim=True
                        )
                    elif not is_interim:
                        final_text = str(frame.data).strip()
                        if not final_text:
                            continue
                        self._last_user_activity_at = time.time()
                        # Ignore duplicate final chunks occasionally emitted by STT.
                        if final_text == self._last_final_text:
                            continue
                        self._last_final_text = final_text
                        self.accumulated_transcript += (
                            " " + final_text
                            if self.accumulated_transcript
                            else final_text
                        )
                        # LiveKit pattern: if VAD already ended before this
                        # transcript arrived (the race condition), trigger EOU
                        # now. VAD speech_ended fired with empty transcript and
                        # bailed — this is the catch-up path.
                        if (
                            not self._vad_speaking
                            and not self.is_responding
                            and (
                                self._debounce_task is None
                                or self._debounce_task.done()
                            )
                        ):
                            logger.info(
                                "Final transcript arrived after VAD ended — scheduling commit: %r",
                                final_text[:60],
                            )
                            self._schedule_turn_commit()

                elif frame.type == FrameType.CONTROL:
                    action = (
                        frame.data.get("action", "")
                        if isinstance(frame.data, dict)
                        else ""
                    )

                    if action == "utterance_end":
                        # utterance_end is used only for transcript accumulation.
                        # Turn commits are driven by VAD speech_ended (primary)
                        # or final transcript arrival after VAD ended (catch-up).
                        transcript = (
                            frame.data.get("transcript", "")
                            if isinstance(frame.data, dict)
                            else ""
                        )
                        text = transcript.strip()
                        if text and text != self._last_final_text:
                            self._last_final_text = text
                            self.accumulated_transcript += (
                                " " + text if self.accumulated_transcript else text
                            )
                        logger.debug(
                            "STT utterance_end (transcript only): %r", transcript[:80]
                        )
                        continue

                    elif action == "speech_started":
                        logger.info(
                            "STT speech_started (responding=%s, assistant_speaking=%s)",
                            self.is_responding,
                            time.time() < self._assistant_speaking_until,
                        )
                        # Only treat STT speech_started as real user activity
                        # when VAD also confirms speech. Deepgram fires
                        # speech_started on background noise constantly.
                        if self._vad_speaking:
                            self._last_user_activity_at = time.time()
                            # User genuinely resumed speaking — cancel any
                            # pending commit so we don't respond mid-sentence.
                            if (
                                not self.is_responding
                                and self._debounce_task
                                and not self._debounce_task.done()
                            ):
                                self._debounce_task.cancel()
                                self._turn_commit_token += 1
                        await self._handle_barge_in(
                            "stt", allow_thinking_interrupt=True
                        )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("STT event loop error: %s", e, exc_info=True)

    async def _debounce_and_trigger(self, token: int) -> None:
        """Wait for silence, then trigger voice response.

        Flow:
        1. Run turn detector ONCE on current transcript to get delay.
        2. If VAD says user is still speaking, wait for speech_ended
           (via cancellation + re-schedule from on_vad_event) rather
           than polling in a hot loop.
        3. After delay elapses, commit if token is still valid and VAD
           is not active. No idle_for guard — VAD is the sole gating
           signal to avoid noise-triggered resets.
        """
        try:
            if token != self._turn_commit_token:
                return
            text = self.accumulated_transcript.strip()
            if not text:
                return

            # If VAD says user is still speaking, bail out now.
            # on_vad_event(speech_ended) will re-schedule when they stop.
            if self._vad_speaking:
                logger.info("Turn commit deferred: VAD reports user still speaking")
                return

            # Run turn detector exactly once for this commit attempt.
            delay_s = await self._resolve_endpoint_delay(text)

            await asyncio.sleep(delay_s)

            if token != self._turn_commit_token:
                return

            # Final VAD guard: if user started speaking again during the
            # delay, bail out — on_vad_event will re-schedule.
            if self._vad_speaking:
                logger.info(
                    "Turn commit aborted after delay: VAD reports user speaking again"
                )
                return

            text = self.accumulated_transcript.strip()
            self.accumulated_transcript = ""
            self._last_final_text = ""
            if not text:
                return

            self._response_task = asyncio.create_task(
                self._trigger_response(text),
                name=f"voice-response-{self.session_id}",
            )
        except asyncio.CancelledError:
            pass  # User still speaking — debounce reset via _schedule_turn_commit

    # ─── Voice Response ──────────────────────────────────────────

    async def _trigger_response(self, text: str) -> None:
        """Complete voice response: LLM → TTS → audio out."""
        self.is_responding = True
        response_start = time.time()
        assistant_text = ""

        try:
            # Send final transcript
            await self._send_text_event("transcript", text, interim=False)
            await self._send_text_event("status", "thinking...")

            # Stream from AgentCore
            sentence_buffer = ""

            async for event in self.agent.generate_reply_voice(text, self.session_id):
                if event.stream_type == "text_chunk":
                    chunk = event.payload.get("text", "")
                    await self._send_text_event("text_chunk", chunk)
                    assistant_text += chunk

                    # Sentence-level TTS for low latency
                    sentence_buffer += chunk
                    sentences = _split_sentences(sentence_buffer)
                    for sentence in sentences[:-1]:
                        if sentence.strip():
                            await self._synthesize_and_send(sentence.strip())
                    sentence_buffer = sentences[-1] if sentences else ""

                elif event.stream_type == "status":
                    # Tool acknowledge — speak it
                    status_text = event.payload.get("message", "")
                    if status_text:
                        await self._send_text_event("status", status_text)
                        await self._synthesize_and_send(status_text)

                elif event.stream_type == "tool_result":
                    await self._send_text_event(
                        "tool_result", json.dumps(event.payload)
                    )

            # Flush remaining text
            if sentence_buffer.strip():
                await self._synthesize_and_send(sentence_buffer.strip())

            await self._send_text_event("stream_end", "")

            self._append_dialog_turn("user", text)
            self._append_dialog_turn("assistant", assistant_text)

            elapsed_ms = (time.time() - response_start) * 1000
            metrics.observe("voice.response_ms", elapsed_ms)

        except asyncio.CancelledError:
            logger.debug("Voice response cancelled (barge-in)")
        except Exception as e:
            logger.error("Voice response error: %s", e, exc_info=True)
            await self._send_text_event("error", str(e))
        finally:
            self.is_responding = False
            if self.is_streaming:
                await self._send_text_event("status", "listening...")

            pending_text = self.accumulated_transcript.strip()
            if pending_text and self.is_streaming:
                logger.info("Processing deferred utterance after response completion")
                self._schedule_turn_commit()

    def _schedule_turn_commit(self) -> None:
        self._turn_commit_token += 1
        token = self._turn_commit_token
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounce_and_trigger(token))

    async def _resolve_endpoint_delay(self, text: str) -> float:
        # In active mode, be conservative on failures to avoid cutting users off.
        default_delay = (
            config.turn_detection.max_endpointing_delay
            if config.turn_detection.mode == "active"
            else 0.5
        )
        if not self._turn_detector:
            return default_delay

        try:
            decision: TurnDecision = await self._turn_detector.predict_delay(
                self._dialog_history,
                text,
                language=None,
            )
        except Exception as e:
            logger.warning(
                "Turn detector inference failed, using default delay: %s (%s)",
                e,
                type(e).__name__,
            )
            return default_delay

        mode = config.turn_detection.mode
        logger.info(
            "Turn detector: p=%.3f threshold=%.3f end=%s delay=%.2fs mode=%s",
            decision.probability,
            decision.threshold,
            decision.likely_end_of_turn,
            decision.recommended_delay_s,
            mode,
        )
        if mode == "active":
            return decision.recommended_delay_s
        return default_delay

    def _append_dialog_turn(self, role: str, content: str) -> None:
        text = content.strip()
        if not text:
            return
        self._dialog_history.append({"role": role, "content": text})
        if len(self._dialog_history) > 12:
            self._dialog_history = self._dialog_history[-12:]

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

        is_assistant_speaking = time.time() < self._assistant_speaking_until
        if not is_assistant_speaking:
            if not allow_thinking_interrupt:
                return

            if not (self._response_task and not self._response_task.done()):
                return

            has_fresh_interim = (
                time.time() - self._last_interim_at < 1.2
                and len(self._last_interim_text.strip()) >= 6
            )
            if not has_fresh_interim:
                return

        self._last_barge_in_at = now

        if probability is not None:
            logger.info(
                "Barge-in (%s, p=%.3f): interrupting assistant playback",
                source,
                probability,
            )
        else:
            logger.info("Barge-in (%s): interrupting assistant playback", source)

        if self.on_barge_in:
            await self.on_barge_in()

        # We interrupted playback, so drop long synthetic cooldown that was
        # based on full TTS length. Keep only a short echo tail guard.
        self._assistant_speaking_until = 0.0
        self._tts_playing = False
        self._tts_cooldown_until = time.time() + 0.35

        if self._response_task and not self._response_task.done():
            self._response_task.cancel()

        self.is_responding = False
        await self.agent.cancel_session(self.session_id)

        if self.is_streaming:
            await self._send_text_event("status", "listening...")

    # ─── TTS ─────────────────────────────────────────────────────

    async def _synthesize_and_send(self, text: str) -> None:
        """Text → TTS → audio out via callback."""
        if not text.strip() or not self.on_audio_out:
            return
        try:
            self._tts_playing = True
            audio_bytes = await self.tts_provider.synthesize(text)
            logger.info(
                "TTS synthesized: %d bytes (%.1fs) for %r",
                len(audio_bytes),
                len(audio_bytes) / 48000.0,
                text[:60],
            )
            await self.on_audio_out(audio_bytes)
            logger.info("TTS audio queued to output track")
            # Estimate playback duration: 24kHz 16-bit mono = 48000 bytes/sec
            playback_secs = len(audio_bytes) / 48000.0
            self._assistant_speaking_until = time.time() + playback_secs
            # Set cooldown: wait for audio to finish playing + 300ms for echo tail
            self._tts_cooldown_until = time.time() + playback_secs + 0.3
        except Exception as e:
            logger.warning("TTS failed: %s", e, exc_info=True)
        finally:
            self._tts_playing = False

    # ─── Event Helpers ───────────────────────────────────────────

    async def _send_text_event(self, event_type: str, data: str, **kwargs: Any) -> None:
        """Send event via data channel callback."""
        if self.on_text_event:
            try:
                await self.on_text_event({"type": event_type, "data": data, **kwargs})
            except Exception as e:
                logger.debug("Failed to send text event %s: %s", event_type, e)

    # ─── Control ─────────────────────────────────────────────────

    def mute(self) -> None:
        self.is_muted = True

    def unmute(self) -> None:
        self.is_muted = False


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries for TTS chunking."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts if parts else [text]
