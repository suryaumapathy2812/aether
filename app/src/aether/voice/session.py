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

from aether.core.frames import Frame, FrameType
from aether.core.metrics import metrics
from aether.providers.deepgram_stt import DeepgramSTTProvider

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
        self.accumulated_transcript = ""
        self._debounce_task: asyncio.Task | None = None
        self._stt_event_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None

        # Callbacks (set by WebRTC transport)
        self.on_audio_out: Callable[[bytes], Awaitable[None]] | None = None
        self.on_text_event: Callable[[dict], Awaitable[None]] | None = None

    # ─── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Called on stream_start. Connect STT, start listening."""
        self.is_streaming = True

        # Start per-session STT
        await self.stt.start()
        await self.stt.connect_stream()
        self._stt_event_task = asyncio.create_task(
            self._stt_event_loop(), name=f"stt-events-{self.session_id}"
        )

        # Greeting (first connection only)
        greeting = await self.agent.generate_greeting()
        if greeting:
            await self._send_text_event("transcript", greeting, role="assistant")
            await self._synthesize_and_send(greeting)
            await self._send_text_event("stream_end", "")

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
        await self.stt.send_audio(pcm_bytes)

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
                        await self._send_text_event(
                            "transcript", frame.data, interim=True
                        )

                elif frame.type == FrameType.CONTROL:
                    action = (
                        frame.data.get("action", "")
                        if isinstance(frame.data, dict)
                        else ""
                    )

                    if action == "utterance_end":
                        transcript = (
                            frame.data.get("transcript", "")
                            if isinstance(frame.data, dict)
                            else ""
                        )
                        self.accumulated_transcript += " " + transcript
                        self.accumulated_transcript = (
                            self.accumulated_transcript.strip()
                        )

                        # Cancel old debounce, start new
                        if self._debounce_task and not self._debounce_task.done():
                            self._debounce_task.cancel()
                        self._debounce_task = asyncio.create_task(
                            self._debounce_and_trigger()
                        )

                    elif action == "speech_started":
                        # Barge-in: cancel in-progress response
                        if self.is_responding and self._response_task:
                            self._response_task.cancel()
                            self.is_responding = False
                            await self.agent.cancel_session(self.session_id)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("STT event loop error: %s", e, exc_info=True)

    async def _debounce_and_trigger(self) -> None:
        """Wait for silence, then trigger voice response."""
        try:
            await asyncio.sleep(0.5)  # 500ms debounce

            text = self.accumulated_transcript.strip()
            self.accumulated_transcript = ""
            if not text:
                return

            self._response_task = asyncio.create_task(
                self._trigger_response(text),
                name=f"voice-response-{self.session_id}",
            )
        except asyncio.CancelledError:
            pass  # User still speaking, debounce reset

    # ─── Voice Response ──────────────────────────────────────────

    async def _trigger_response(self, text: str) -> None:
        """Complete voice response: LLM → TTS → audio out."""
        self.is_responding = True
        response_start = time.time()

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

    # ─── TTS ─────────────────────────────────────────────────────

    async def _synthesize_and_send(self, text: str) -> None:
        """Text → TTS → audio out via callback."""
        if not text.strip() or not self.on_audio_out:
            return
        try:
            audio_bytes = await self.tts_provider.synthesize(text)
            await self.on_audio_out(audio_bytes)
        except Exception as e:
            logger.warning("TTS failed: %s", e)

    # ─── Event Helpers ───────────────────────────────────────────

    async def _send_text_event(self, event_type: str, data: str, **kwargs: Any) -> None:
        """Send event via data channel callback."""
        if self.on_text_event:
            try:
                await self.on_text_event({"type": event_type, "data": data, **kwargs})
            except Exception:
                pass  # Data channel may be closed

    # ─── Control ─────────────────────────────────────────────────

    def mute(self) -> None:
        self.is_muted = True

    def unmute(self) -> None:
        self.is_muted = False


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries for TTS chunking."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts if parts else [text]
