"""
STT Processor — Deepgram speech-to-text.

v0.02: Supports both batch (file) and live streaming transcription.

In streaming mode, the processor manages a persistent Deepgram WebSocket
connection. Audio chunks flow in continuously, interim transcripts come back
in real-time, and an UtteranceEnd event signals the user stopped speaking.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncGenerator

from deepgram import AsyncDeepgramClient

from aether.core.frames import Frame, FrameType, text_frame, control_frame
from aether.core.processor import Processor

logger = logging.getLogger(__name__)


class STTProcessor(Processor):
    """Batch STT — transcribes a complete audio buffer. Used in pre-pipeline."""

    def __init__(self):
        super().__init__("STT")
        self.client: AsyncDeepgramClient | None = None

    async def start(self) -> None:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        self.client = AsyncDeepgramClient(api_key=api_key)
        logger.info("STT processor ready (Deepgram)")

    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Transcribe a complete audio buffer to text."""
        if frame.type != FrameType.AUDIO:
            yield frame
            return

        if not self.client:
            raise RuntimeError("STT processor not started")

        try:
            response = await self.client.listen.v1.media.transcribe_file(
                request=frame.data,
                model="nova-3",
                smart_format=True,
                language="en",
            )

            transcript = response.results.channels[0].alternatives[0].transcript

            if transcript and transcript.strip():
                logger.info(f"STT: '{transcript[:80]}'")
                yield text_frame(transcript.strip(), role="user")
            else:
                logger.debug("STT: empty transcription, skipping")

        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)


class StreamingSTTProcessor:
    """
    Live streaming STT — maintains a persistent Deepgram WebSocket connection.

    Audio chunks are sent continuously. The processor yields:
    - text_frame with interim=True for partial transcripts (so client can display live)
    - text_frame with interim=False for final transcripts (confirmed words)
    - control_frame("utterance_end") when the user stops speaking

    This is NOT a Processor subclass because it doesn't fit the frame-in/frame-out
    model — it manages its own WebSocket lifecycle and emits events asynchronously.
    It's used directly by main.py.
    """

    def __init__(self):
        self.client: AsyncDeepgramClient | None = None
        self._socket = None
        self._transcript_buffer: str = (
            ""  # Accumulates final transcripts within an utterance
        )
        self._interim_text: str = ""  # Latest interim (partial) transcript

        # Queues for communicating between Deepgram's listener and our consumer
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._listen_task: asyncio.Task | None = None

    async def start(self) -> None:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        self.client = AsyncDeepgramClient(api_key=api_key)
        logger.info("Streaming STT ready (Deepgram live)")

    async def connect(self) -> None:
        """Open the Deepgram live WebSocket connection."""
        if not self.client:
            raise RuntimeError("Streaming STT not started")

        self._socket_ctx = self.client.listen.v1.connect(
            model="nova-3",
            language="en",
            smart_format="true",
            interim_results="true",
            utterance_end_ms="1200",  # 1.2s silence = utterance end
            vad_events="true",
            endpointing="300",  # 300ms endpointing for faster response
            encoding="linear16",
            sample_rate="16000",
        )
        self._socket = await self._socket_ctx.__aenter__()

        # Start background task to listen for responses
        self._listen_task = asyncio.create_task(self._listen_loop())
        logger.info("Deepgram live connection opened")

    async def disconnect(self) -> None:
        """Close the Deepgram connection."""
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

        logger.info("Deepgram live connection closed")

    async def send_audio(self, audio_data: bytes) -> None:
        """Send an audio chunk to Deepgram for real-time transcription."""
        if self._socket:
            try:
                await self._socket.send_media(audio_data)
            except Exception as e:
                logger.error(f"Error sending audio to Deepgram: {e}")

    async def events(self) -> AsyncGenerator[Frame, None]:
        """
        Yield frames as Deepgram produces transcription events.

        Yields:
        - text_frame(role="user", interim=True) for partial transcripts
        - text_frame(role="user", interim=False) for final transcript segments
        - control_frame("utterance_end") when user stops speaking
        """
        while True:
            try:
                frame = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield frame
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _listen_loop(self) -> None:
        """Background task: listen to Deepgram WebSocket and push events to queue."""
        if not self._socket:
            return

        try:
            async for message in self._socket:
                await self._handle_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Deepgram listen error: {e}", exc_info=True)

    async def _handle_message(self, message) -> None:
        """Route Deepgram messages to appropriate handlers."""
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
                # Final transcript segment — confirmed words
                self._transcript_buffer += (
                    " " + transcript.strip() if self._transcript_buffer else transcript
                )
                self._interim_text = ""

                await self._event_queue.put(text_frame(transcript, role="user"))
                # Add metadata to indicate this is a final segment
                logger.info(f"STT final: '{transcript[:60]}'")

                if speech_final:
                    # speech_final means this utterance is complete
                    await self._emit_utterance_end()
            else:
                # Interim result — partial, may change
                self._interim_text = transcript
                interim_frame = text_frame(transcript, role="user")
                interim_frame.metadata["interim"] = True
                await self._event_queue.put(interim_frame)
                logger.debug(f"STT interim: '{transcript[:60]}'")

        elif msg_type == "UtteranceEnd":
            await self._emit_utterance_end()

        elif msg_type == "SpeechStarted":
            logger.debug("Speech detected")
            await self._event_queue.put(control_frame("speech_started"))

    async def _emit_utterance_end(self) -> None:
        """Emit the accumulated transcript as a complete utterance."""
        final_text = self._transcript_buffer.strip()
        if final_text:
            logger.info(f"STT utterance complete: '{final_text[:80]}'")
            # Emit a control frame with the complete utterance text
            await self._event_queue.put(
                control_frame("utterance_end", transcript=final_text)
            )
        self._transcript_buffer = ""
        self._interim_text = ""
