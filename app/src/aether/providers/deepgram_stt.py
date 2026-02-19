"""
Deepgram STT Provider — batch and live streaming transcription.

v0.04: Rewritten for Deepgram SDK v5 event-based API.
Uses connection.on(EventType.MESSAGE, handler) + start_listening()
per the official SDK examples.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    ListenV1MetadataEvent,
    ListenV1ResultsEvent,
    ListenV1SpeechStartedEvent,
    ListenV1UtteranceEndEvent,
)

from aether.core.config import config
from aether.core.frames import Frame, text_frame, control_frame
from aether.core.metrics import metrics
from aether.providers.base import STTProvider

logger = logging.getLogger(__name__)


class DeepgramSTTProvider(STTProvider):
    def __init__(self):
        self.client: AsyncDeepgramClient | None = None
        self._socket = None
        self._socket_ctx = None
        self._transcript_buffer: str = ""
        self._interim_text: str = ""
        self._event_queue: asyncio.Queue[Frame] = asyncio.Queue()
        self._listen_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._connected = False
        self._reconnecting = False
        self._should_be_connected = False
        self._audio_chunks_sent = 0
        self._last_send_at: float = 0.0

    async def start(self) -> None:
        if self.client:
            return
        cfg = config.stt
        if not cfg.api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        self.client = AsyncDeepgramClient(api_key=cfg.api_key)
        logger.info("Deepgram STT ready (model=%s)", cfg.model)

    async def stop(self) -> None:
        self._should_be_connected = False
        await self.disconnect_stream()
        self.client = None

    async def transcribe(self, audio_data: bytes) -> str | None:
        """Batch transcription of a complete audio buffer."""
        if not self.client:
            raise RuntimeError("Deepgram STT not started")
        try:
            response = await self.client.listen.v1.media.transcribe_file(
                request=audio_data,
                model=config.stt.model,
                smart_format=True,
                language=config.stt.language,
            )
            transcript = response.results.channels[0].alternatives[0].transcript
            if transcript and transcript.strip():
                return transcript.strip()
            return None
        except Exception as e:
            logger.error("Deepgram batch transcription error: %s", e, exc_info=True)
            return None

    # ─── Live Streaming ──────────────────────────────────────────

    async def connect_stream(self) -> None:
        """Open live WebSocket connection to Deepgram."""
        if not self.client:
            raise RuntimeError("Deepgram STT not started")
        self._should_be_connected = True
        self._audio_chunks_sent = 0
        metrics.inc("provider.stt.connections")
        await self._do_connect()

    async def _do_connect(self) -> None:
        """Internal connect — used by both initial connect and reconnect."""
        cfg = config.stt
        try:
            self._socket_ctx = self.client.listen.v1.connect(
                model=cfg.model,
                language=cfg.language,
                channels="1",
                smart_format="true",
                interim_results="true",
                utterance_end_ms=str(cfg.utterance_end_ms),
                vad_events="true",
                endpointing=str(cfg.endpointing_ms),
                encoding=cfg.encoding,
                sample_rate=str(cfg.sample_rate),
            )
            self._socket = await self._socket_ctx.__aenter__()

            # Register v5 event handlers
            self._socket.on(EventType.MESSAGE, self._on_message)
            self._socket.on(EventType.ERROR, self._on_error)
            self._socket.on(EventType.CLOSE, self._on_close)

            # start_listening() runs the websocket receive loop and
            # dispatches parsed messages to the callbacks above.
            self._listen_task = asyncio.create_task(
                self._socket.start_listening(),
                name="deepgram-listen",
            )

            # Log if the listen task dies unexpectedly
            def _on_listen_done(task: asyncio.Task) -> None:
                if task.cancelled():
                    logger.debug("Deepgram listen task cancelled")
                elif task.exception():
                    logger.error(
                        "Deepgram listen task died: %s",
                        task.exception(),
                        exc_info=task.exception(),
                    )
                else:
                    logger.info("Deepgram listen task finished normally")

            self._listen_task.add_done_callback(_on_listen_done)

            # Keepalive loop prevents net0001 timeouts during long silence/
            # suppression windows by sending protocol keepalive frames.
            self._keepalive_task = asyncio.create_task(
                self._keepalive_loop(),
                name="deepgram-keepalive",
            )

            self._connected = True
            self._transcript_buffer = ""
            self._interim_text = ""
            self._last_send_at = time.monotonic()
            logger.info("Deepgram live connection opened")

        except Exception as e:
            self._connected = False
            logger.error("Deepgram connect failed: %s", e, exc_info=True)
            raise

    async def disconnect_stream(self) -> None:
        """Close the Deepgram live connection."""
        self._should_be_connected = False
        self._connected = False

        # Cancel the start_listening task
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listen_task = None

        # Cancel keepalive task
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._keepalive_task = None

        # Send CloseStream control and exit context manager
        if self._socket:
            try:
                from deepgram.extensions.types.sockets import ListenV1ControlMessage

                await self._socket.send_control(
                    ListenV1ControlMessage(type="CloseStream")
                )
            except Exception:
                pass
            try:
                await self._socket_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._socket = None
            self._socket_ctx = None

        logger.info("Deepgram live connection closed")

    async def send_audio(self, chunk: bytes) -> None:
        """Send audio chunk to Deepgram. Triggers reconnect on failure."""
        if self._reconnecting:
            return  # Drop audio during reconnect

        if self._socket and self._connected:
            try:
                self._audio_chunks_sent += 1
                if self._audio_chunks_sent <= 3:
                    # Log first few chunks for debugging audio format
                    import numpy as np

                    samples = np.frombuffer(chunk, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                    logger.debug(
                        "Audio chunk #%d to Deepgram: %d bytes, %d samples, "
                        "rms=%.1f, min=%d, max=%d, first_10=%s",
                        self._audio_chunks_sent,
                        len(chunk),
                        len(samples),
                        rms,
                        int(samples.min()),
                        int(samples.max()),
                        samples[:10].tolist(),
                    )
                elif self._audio_chunks_sent % 500 == 0:
                    logger.debug(
                        "Deepgram audio chunks sent: %d", self._audio_chunks_sent
                    )
                await self._socket.send_media(chunk)
                self._last_send_at = time.monotonic()
            except Exception as e:
                logger.error("Send audio failed: %s", e)
                if self._should_be_connected:
                    asyncio.create_task(self._reconnect())
        else:
            # Not connected yet — drop silently (happens during startup)
            pass

    async def stream_events(self) -> AsyncGenerator[Frame, None]:
        """Yield frames as Deepgram produces transcription events."""
        while True:
            try:
                frame = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield frame
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def is_connected(self) -> bool:
        return self._connected and self._socket is not None

    async def health_check(self) -> dict:
        return {
            "provider": "deepgram",
            "status": "connected" if self._connected else "disconnected",
            "streaming": self._should_be_connected,
            "reconnecting": self._reconnecting,
        }

    # ─── V5 Event Callbacks ──────────────────────────────────────
    # These are called by start_listening() when messages arrive.
    # They run in the same event loop, so we can put to the queue directly.

    def _on_message(self, message) -> None:
        """Handle parsed Deepgram message (called by start_listening)."""
        logger.debug("Deepgram event: %s", type(message).__name__)
        # start_listening dispatches parsed Pydantic models.
        # Route by isinstance check (more robust than string type check).
        if isinstance(message, ListenV1ResultsEvent):
            self._handle_results(message)
        elif isinstance(message, ListenV1UtteranceEndEvent):
            self._handle_utterance_end()
        elif isinstance(message, ListenV1SpeechStartedEvent):
            logger.debug("Speech detected")
            self._event_queue.put_nowait(control_frame("speech_started"))
        elif isinstance(message, ListenV1MetadataEvent):
            logger.debug(
                "Deepgram metadata: request_id=%s", getattr(message, "request_id", "?")
            )
        else:
            logger.debug("Deepgram unknown message: %s", type(message).__name__)

    def _on_error(self, error) -> None:
        """Handle Deepgram WebSocket error."""
        logger.error("Deepgram WS error: %s", error)
        if self._should_be_connected and not self._reconnecting:
            asyncio.create_task(self._reconnect())

    def _on_close(self, _) -> None:
        """Handle Deepgram WebSocket close."""
        logger.info("Deepgram WS closed")
        self._connected = False
        if self._should_be_connected and not self._reconnecting:
            asyncio.create_task(self._reconnect())

    # ─── Message Handlers ────────────────────────────────────────

    def _handle_results(self, message: ListenV1ResultsEvent) -> None:
        """Process a Results event — interim or final transcript."""
        channel = message.channel
        if not channel or not channel.alternatives:
            logger.debug("Results: no channel/alternatives")
            return

        transcript = channel.alternatives[0].transcript
        is_final = message.is_final or False
        speech_final = message.speech_final or False

        logger.debug(
            "Results: transcript=%r, is_final=%s, speech_final=%s",
            transcript[:80] if transcript else "(empty)",
            is_final,
            speech_final,
        )

        if not transcript:
            return

        if is_final:
            self._transcript_buffer += (
                " " + transcript.strip() if self._transcript_buffer else transcript
            )
            self._interim_text = ""
            self._event_queue.put_nowait(text_frame(transcript, role="user"))
            logger.debug("STT final: '%s'", transcript[:60])

            if speech_final:
                self._handle_utterance_end()
        else:
            self._interim_text = transcript
            interim = text_frame(transcript, role="user")
            interim.metadata["interim"] = True
            self._event_queue.put_nowait(interim)
            logger.debug("STT interim: '%s'", transcript[:60])

    def _handle_utterance_end(self) -> None:
        """Emit accumulated transcript as a complete utterance."""
        final_text = self._transcript_buffer.strip()
        if final_text:
            logger.debug("STT utterance: '%s'", final_text[:80])
            word_count = len(final_text.split())
            estimated_duration_ms = (word_count / 150) * 60 * 1000
            metrics.observe("provider.stt.utterance_duration_ms", estimated_duration_ms)
            self._event_queue.put_nowait(
                control_frame("utterance_end", transcript=final_text)
            )
        self._transcript_buffer = ""
        self._interim_text = ""

    # ─── Auto-reconnect ──────────────────────────────────────────

    async def _keepalive_loop(self) -> None:
        """Send Deepgram keepalive frames when media is idle."""
        try:
            while self._should_be_connected:
                await asyncio.sleep(2.0)
                if not self._connected or not self._socket or self._reconnecting:
                    continue

                idle_for = time.monotonic() - self._last_send_at
                if idle_for < 4.0:
                    continue

                try:
                    await self._socket.send_keep_alive()
                    self._last_send_at = time.monotonic()
                    logger.debug("Deepgram keepalive sent (idle_for=%.1fs)", idle_for)
                except Exception as e:
                    logger.warning("Deepgram keepalive failed: %s", e)
                    if self._should_be_connected and not self._reconnecting:
                        asyncio.create_task(self._reconnect())
        except asyncio.CancelledError:
            pass

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        if self._reconnecting:
            return
        self._reconnecting = True
        self._connected = False

        cfg = config.stt
        delay = cfg.reconnect_delay

        # Clean up old connection
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listen_task = None

        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._keepalive_task = None

        if self._socket:
            try:
                await self._socket_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._socket = None
            self._socket_ctx = None

        for attempt in range(1, cfg.reconnect_attempts + 1):
            if not self._should_be_connected:
                logger.info("Reconnect cancelled — stream no longer needed")
                break

            logger.info(
                "Deepgram reconnect attempt %d/%d (delay: %.1fs)",
                attempt,
                cfg.reconnect_attempts,
                delay,
            )
            await asyncio.sleep(delay)

            try:
                await self._do_connect()
                logger.info("Deepgram reconnected on attempt %d", attempt)
                self._event_queue.put_nowait(control_frame("reconnected"))
                self._reconnecting = False
                return
            except Exception as e:
                logger.warning("Reconnect attempt %d failed: %s", attempt, e)
                delay = min(delay * 2, 30.0)

        logger.error("Deepgram reconnect failed — all attempts exhausted")
        self._event_queue.put_nowait(control_frame("connection_lost"))
        self._reconnecting = False
