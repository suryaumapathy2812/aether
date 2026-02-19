"""
Deepgram STT Provider — batch and live streaming transcription.

v0.03: Auto-reconnect on WebSocket drop. If Deepgram disconnects mid-conversation,
we reconnect transparently. The user never knows.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from deepgram import AsyncDeepgramClient

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
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._listen_task: asyncio.Task | None = None
        self._connected = False
        self._reconnecting = False
        self._should_be_connected = (
            False  # Intent flag — are we supposed to be streaming?
        )

    async def start(self) -> None:
        if self.client:
            return  # Already started
        cfg = config.stt
        if not cfg.api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        self.client = AsyncDeepgramClient(api_key=cfg.api_key)
        logger.info("Deepgram STT ready")

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
            logger.error(f"Deepgram batch transcription error: {e}", exc_info=True)
            return None

    async def connect_stream(self) -> None:
        """Open live WebSocket connection to Deepgram."""
        if not self.client:
            raise RuntimeError("Deepgram STT not started")

        self._should_be_connected = True
        metrics.inc("provider.stt.connections")
        await self._do_connect()

    async def _do_connect(self) -> None:
        """Internal connect — used by both initial connect and reconnect."""
        cfg = config.stt

        try:
            self._socket_ctx = self.client.listen.v1.connect(
                model=cfg.model,
                language=cfg.language,
                smart_format="true",
                interim_results="true",
                utterance_end_ms=str(cfg.utterance_end_ms),
                vad_events="true",
                endpointing=str(cfg.endpointing_ms),
                encoding=cfg.encoding,
                sample_rate=str(cfg.sample_rate),
            )
            self._socket = await self._socket_ctx.__aenter__()
            self._connected = True
            self._transcript_buffer = ""
            self._interim_text = ""

            # Start listening
            self._listen_task = asyncio.create_task(self._listen_loop())
            logger.debug("Deepgram live connection opened")

        except Exception as e:
            self._connected = False
            logger.error(f"Deepgram connect failed: {e}", exc_info=True)
            raise

    async def disconnect_stream(self) -> None:
        """Close the Deepgram live connection."""
        self._should_be_connected = False
        self._connected = False

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

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

        logger.debug("Deepgram live connection closed")

    async def send_audio(self, chunk: bytes) -> None:
        """Send audio chunk. If connection is dead, trigger reconnect."""
        if self._reconnecting:
            return  # Drop audio during reconnect — it's brief

        if self._socket and self._connected:
            try:
                await self._socket.send_media(chunk)
            except Exception as e:
                logger.error(f"Send audio failed: {e}")
                if self._should_be_connected:
                    asyncio.create_task(self._reconnect())

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

    # --- Auto-reconnect ---

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        if self._reconnecting:
            return  # Already reconnecting
        self._reconnecting = True
        self._connected = False

        cfg = config.stt
        delay = cfg.reconnect_delay

        # Clean up old connection
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._socket:
            try:
                await self._socket_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._socket = None

        for attempt in range(1, cfg.reconnect_attempts + 1):
            if not self._should_be_connected:
                logger.info("Reconnect cancelled — stream no longer needed")
                break

            logger.debug(
                f"Deepgram reconnect attempt {attempt}/{cfg.reconnect_attempts} (delay: {delay:.1f}s)"
            )
            await asyncio.sleep(delay)

            try:
                await self._do_connect()
                logger.info(f"Deepgram reconnected on attempt {attempt}")
                # Notify client that we're back
                await self._event_queue.put(control_frame("reconnected"))
                self._reconnecting = False
                return
            except Exception as e:
                logger.warning(f"Reconnect attempt {attempt} failed: {e}")
                delay = min(delay * 2, 30.0)  # Exponential backoff, cap at 30s

        logger.error("Deepgram reconnect failed — all attempts exhausted")
        await self._event_queue.put(control_frame("connection_lost"))
        self._reconnecting = False

    # --- Internal listener ---

    async def _listen_loop(self) -> None:
        """Background task: listen to Deepgram WebSocket."""
        if not self._socket:
            return

        try:
            async for message in self._socket:
                await self._handle_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Deepgram listen error: {e}", exc_info=True)
            # Connection died — try to reconnect
            if self._should_be_connected and not self._reconnecting:
                asyncio.create_task(self._reconnect())

    async def _handle_message(self, message) -> None:
        """Route Deepgram messages."""
        msg_type = getattr(message, "type", None)

        if msg_type == "Results":
            channel = message.channel
            if not channel or not channel.alternatives:
                return

            transcript = channel.alternatives[0].transcript
            is_final = getattr(message, "is_final", False)
            speech_final = getattr(message, "speech_final", False)

            if not transcript:
                return

            if is_final:
                self._transcript_buffer += (
                    " " + transcript.strip() if self._transcript_buffer else transcript
                )
                self._interim_text = ""
                await self._event_queue.put(text_frame(transcript, role="user"))
                logger.debug(f"STT final: '{transcript[:60]}'")

                if speech_final:
                    await self._emit_utterance_end()
            else:
                self._interim_text = transcript
                interim = text_frame(transcript, role="user")
                interim.metadata["interim"] = True
                await self._event_queue.put(interim)
                logger.debug(f"STT interim: '{transcript[:60]}'")

        elif msg_type == "UtteranceEnd":
            await self._emit_utterance_end()

        elif msg_type == "SpeechStarted":
            logger.debug("Speech detected")
            await self._event_queue.put(control_frame("speech_started"))

    async def _emit_utterance_end(self) -> None:
        """Emit accumulated transcript as a complete utterance."""
        final_text = self._transcript_buffer.strip()
        if final_text:
            logger.debug(f"STT utterance: '{final_text[:80]}'")
            # Estimate utterance duration from word count (avg ~150 wpm)
            word_count = len(final_text.split())
            estimated_duration_ms = (word_count / 150) * 60 * 1000
            metrics.observe("provider.stt.utterance_duration_ms", estimated_duration_ms)
            await self._event_queue.put(
                control_frame("utterance_end", transcript=final_text)
            )
        self._transcript_buffer = ""
        self._interim_text = ""
