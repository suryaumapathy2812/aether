"""
Voice Adapter — handles audio I/O with STT/TTS.

Orchestrates the full voice pipeline:
- STT streaming with debounce + utterance detection
- Memory retrieval
- LLM generation via LLMCore
- TTS synthesis for assistant responses

This extracts the voice-specific logic from CoreHandler:
- _stt_event_loop
- _debounce_and_trigger
- _trigger_voice_response
- _run_voice_pipeline (TTS portion)
- _session_greeting
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

import aether.core.config as config_module
from aether.core.frames import FrameType, text_frame
from aether.kernel.contracts import JobModality
from aether.modality.base import ModalityAdapter

if TYPE_CHECKING:
    from aether.greeting import generate_greeting
    from aether.llm.context_builder import ContextBuilder
    from aether.llm.core import LLMCore
    from aether.memory.store import MemoryStore
    from aether.processors.tts import TTSProcessor
    from aether.providers.base import LLMProvider, STTProvider, TTSProvider
    from aether.transport.core_msg import CoreMsg

logger = logging.getLogger(__name__)


class VoiceAdapter(ModalityAdapter):
    """
    Voice modality adapter — STT input, TTS output.

    Handles:
    - Streaming STT event loop with debounce
    - Utterance detection and accumulation
    - LLM pipeline (via LLMCore)
    - TTS synthesis for each sentence
    - Session greeting on connect
    - Mute/unmute state
    - Half-duplex: drops STT events while responding
    """

    def __init__(
        self,
        llm_core: "LLMCore",
        context_builder: "ContextBuilder",
        memory_store: "MemoryStore",
        stt_provider: "STTProvider",
        tts_provider: "TTSProvider",
        llm_provider: "LLMProvider",
    ) -> None:
        self._llm_core = llm_core
        self._context_builder = context_builder
        self._memory_store = memory_store
        self._stt_provider = stt_provider
        self._tts_provider = tts_provider
        self._llm_provider = llm_provider

        # TTS processor (lazily created per session)
        self._tts_processors: dict[str, "TTSProcessor"] = {}

    @property
    def modality(self) -> str:
        return JobModality.VOICE.value

    async def handle_input(
        self,
        msg: "CoreMsg",
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """
        Process voice input through the full pipeline.

        Routes based on content type:
        - TextContent: direct text (already transcribed) → LLM + TTS
        - AudioContent: batch audio → STT → LLM + TTS
        - EventContent: stream_start, audio_chunk, mute, etc.
        """
        from aether.transport.core_msg import (
            AudioContent,
            CoreMsg,
            EventContent,
            TextContent,
        )

        content = msg.content

        if isinstance(content, TextContent):
            async for resp in self._handle_text_input(msg, session_state):
                yield resp

        elif isinstance(content, AudioContent):
            async for resp in self._handle_audio_input(msg, session_state):
                yield resp

        elif isinstance(content, EventContent):
            async for resp in self._handle_event(msg, session_state):
                yield resp

    async def handle_output(
        self,
        event_type: str,
        payload: dict[str, Any],
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """Convert kernel events to voice transport messages (text + audio)."""
        from aether.transport.core_msg import CoreMsg

        user_id = session_state.get("user_id", "")
        session_id = session_state.get("session_id", "")

        if event_type == "text_chunk":
            # Text chunk for display
            yield CoreMsg.text(
                text=payload.get("text", ""),
                user_id=user_id,
                session_id=session_id,
                role=payload.get("role", "assistant"),
                transport="text_chunk",
            )
            # Also synthesize TTS
            tts = self._get_tts(session_id)
            if tts:
                async for audio_msg in self._synthesize(
                    payload.get("text", ""), user_id, session_id, tts
                ):
                    yield audio_msg

        elif event_type == "status":
            yield CoreMsg.text(
                text=payload.get("text", ""),
                user_id=user_id,
                session_id=session_id,
                role="system",
                transport="status",
            )

        elif event_type == "stream_end":
            yield CoreMsg.event(
                event_type="stream_end",
                user_id=user_id,
                session_id=session_id,
                transport="control",
            )

    async def on_session_start(self, session_state: dict[str, Any]) -> None:
        """Initialize TTS processor for this session."""
        from aether.processors.tts import TTSProcessor

        session_id = session_state.get("session_id", "")
        if session_id and session_id not in self._tts_processors:
            tts = TTSProcessor(self._tts_provider)
            await tts.start()
            self._tts_processors[session_id] = tts

    async def on_session_end(self, session_state: dict[str, Any]) -> None:
        """Cleanup TTS processor for this session."""
        session_id = session_state.get("session_id", "")
        tts = self._tts_processors.pop(session_id, None)
        if tts:
            await tts.stop()

    # ─── Text Input (already transcribed) ────────────────────────

    async def _handle_text_input(
        self,
        msg: "CoreMsg",
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """Handle pre-transcribed text → LLM + TTS pipeline."""
        from aether.transport.core_msg import CoreMsg, TextContent

        content: TextContent = msg.content  # type: ignore[assignment]
        user_text = content.text
        user_id = msg.user_id
        session_id = msg.session_id

        session_state.setdefault("turn_count", 0)
        session_state["turn_count"] += 1

        async for resp in self._run_voice_pipeline(
            user_text, user_id, session_id, session_state
        ):
            yield resp

    # ─── Audio Input (batch STT) ─────────────────────────────────

    async def _handle_audio_input(
        self,
        msg: "CoreMsg",
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """Handle batch audio → STT → LLM + TTS pipeline."""
        from aether.transport.core_msg import AudioContent, CoreMsg

        content: AudioContent = msg.content  # type: ignore[assignment]
        user_text = await self._stt_provider.transcribe(content.audio_data)

        if user_text:
            yield CoreMsg.text(
                text=user_text,
                user_id=msg.user_id,
                session_id=msg.session_id,
                role="system",
                transport="transcript",
            )

            session_state.setdefault("turn_count", 0)
            session_state["turn_count"] += 1

            async for resp in self._run_voice_pipeline(
                user_text, msg.user_id, msg.session_id, session_state
            ):
                yield resp
        else:
            yield CoreMsg.text(
                text="Didn't catch that, try again",
                user_id=msg.user_id,
                session_id=msg.session_id,
                role="system",
                transport="status",
            )

    # ─── Event Handling ──────────────────────────────────────────

    async def _handle_event(
        self,
        msg: "CoreMsg",
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """Handle voice events: stream_start, audio_chunk, mute, etc."""
        from aether.transport.core_msg import CoreMsg, EventContent

        content: EventContent = msg.content  # type: ignore[assignment]
        event_type = content.event_type
        user_id = msg.user_id
        session_id = msg.session_id

        if event_type == "stream_start":
            await self._stt_provider.connect_stream()

            # Start STT event listener
            callback = session_state.get("background_callback")
            if callback:
                task = asyncio.create_task(
                    self._stt_event_loop(user_id, session_id, session_state, callback)
                )
                session_state["stt_event_task"] = task

            yield CoreMsg.text(
                text="listening...",
                user_id=user_id,
                session_id=session_id,
                role="system",
                transport="status",
            )

            # Session greeting
            is_reconnect = content.payload.get("reconnect", False)
            if not is_reconnect:
                async for resp in self._session_greeting(
                    user_id, session_id, session_state
                ):
                    yield resp

        elif event_type == "stream_stop":
            task = session_state.get("stt_event_task")
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                session_state["stt_event_task"] = None
            await self._stt_provider.disconnect_stream()

        elif event_type == "audio_chunk":
            audio_b64 = content.payload.get("data", "")
            is_muted = session_state.get("is_muted", False)
            if audio_b64 and not is_muted:
                audio_data = base64.b64decode(audio_b64)
                await self._stt_provider.send_audio(audio_data)

        elif event_type == "mute":
            session_state["is_muted"] = True
            debounce_task = session_state.get("debounce_task")
            if debounce_task and not debounce_task.done():
                debounce_task.cancel()
            session_state["accumulated_transcript"] = ""
            yield CoreMsg.text(
                text="muted",
                user_id=user_id,
                session_id=session_id,
                role="system",
                transport="status",
            )

        elif event_type == "unmute":
            session_state["is_muted"] = False
            yield CoreMsg.text(
                text="listening...",
                user_id=user_id,
                session_id=session_id,
                role="system",
                transport="status",
            )

        elif event_type == "image":
            image_b64 = content.payload.get("data", "")
            mime = content.payload.get("mime", "image/jpeg")
            if image_b64:
                session_state["pending_vision"] = {
                    "data": base64.b64decode(image_b64),
                    "mime_type": mime,
                }
            yield CoreMsg.text(
                text="Image received, listening...",
                user_id=user_id,
                session_id=session_id,
                role="system",
                transport="status",
            )

    # ─── Voice Pipeline ──────────────────────────────────────────

    async def _run_voice_pipeline(
        self,
        user_text: str,
        user_id: str,
        session_id: str,
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """Memory → LLM → TTS streaming pipeline."""
        from aether.llm.context_builder import SessionState
        from aether.llm.contracts import LLMEventType
        from aether.transport.core_msg import CoreMsg

        tts = self._get_tts(session_id)

        # Build session for context builder
        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            mode="voice",
            history=session_state.get("history", []),
        )

        # Memory retrieval
        pending_memory = await self._retrieve_memory(user_text)

        # Vision context
        pending_vision = session_state.pop("pending_vision", None)

        # Build LLM request
        envelope = await self._context_builder.build(
            user_message=user_text,
            session=session,
            enabled_plugins=session_state.get("enabled_plugins", []),
            pending_memory=pending_memory,
            pending_vision=pending_vision,
        )

        # Stream LLM response with TTS
        sentence_index = 0

        async for event in self._llm_core.generate_with_tools(envelope):
            if event.event_type == LLMEventType.TEXT_CHUNK.value:
                text = event.payload.get("text", "")

                # Text chunk for display
                yield CoreMsg.text(
                    text=text,
                    user_id=user_id,
                    session_id=session_id,
                    role="assistant",
                    transport="text_chunk",
                    sentence_index=sentence_index,
                )

                # TTS synthesis
                if tts:
                    async for audio_msg in self._synthesize(
                        text, user_id, session_id, tts, sentence_index
                    ):
                        yield audio_msg

                sentence_index += 1

            elif event.event_type == LLMEventType.TOOL_RESULT.value:
                yield CoreMsg.event(
                    event_type="tool_result",
                    user_id=user_id,
                    session_id=session_id,
                    payload={
                        "name": event.payload.get("tool_name", "unknown"),
                        "output": event.payload.get("output", "")[:500],
                        "error": event.payload.get("error", False),
                    },
                    transport="tools",
                )

                # Speak status for tool use
                status_text = f"Working on that..."
                yield CoreMsg.text(
                    text=status_text,
                    user_id=user_id,
                    session_id=session_id,
                    role="system",
                    transport="status",
                )

            elif event.event_type == LLMEventType.STREAM_END.value:
                yield CoreMsg.event(
                    event_type="stream_end",
                    user_id=user_id,
                    session_id=session_id,
                    transport="control",
                )

            elif event.event_type == LLMEventType.ERROR.value:
                yield CoreMsg.text(
                    text=event.payload.get("message", "Something went wrong."),
                    user_id=user_id,
                    session_id=session_id,
                    role="system",
                    transport="status",
                )

        logger.info(f"[voice] {sentence_index} sentences")

    # ─── STT Event Loop ──────────────────────────────────────────

    async def _stt_event_loop(
        self,
        user_id: str,
        session_id: str,
        session_state: dict[str, Any],
        callback: Any,
    ) -> None:
        """
        Background task: listen to streaming STT events.
        Half-duplex: drops events while responding or muted.
        """
        try:
            async for event in self._stt_provider.stream_events():
                is_responding = session_state.get("is_responding", False)
                is_muted = session_state.get("is_muted", False)

                if is_responding or is_muted:
                    continue

                if event.type == FrameType.TEXT and event.metadata.get("interim"):
                    from aether.transport.core_msg import CoreMsg

                    msg = CoreMsg.text(
                        text=event.data,
                        user_id=user_id,
                        session_id=session_id,
                        role="system",
                        transport="transcript_interim",
                    )
                    await callback(msg)

                elif event.type == FrameType.CONTROL:
                    action = event.data.get("action")

                    if action == "utterance_end":
                        transcript = event.data.get("transcript", "")
                        if transcript:
                            accumulated = session_state.get(
                                "accumulated_transcript", ""
                            )
                            session_state["accumulated_transcript"] = (
                                accumulated + " " + transcript
                                if accumulated
                                else transcript
                            )
                            # Reset debounce
                            debounce_task = session_state.get("debounce_task")
                            if debounce_task and not debounce_task.done():
                                debounce_task.cancel()
                            session_state["debounce_task"] = asyncio.create_task(
                                self._debounce_and_trigger(
                                    user_id, session_id, session_state, callback
                                )
                            )

                    elif action == "reconnected":
                        from aether.transport.core_msg import CoreMsg

                        msg = CoreMsg.text(
                            text="listening...",
                            user_id=user_id,
                            session_id=session_id,
                            role="system",
                            transport="status",
                        )
                        await callback(msg)
                        logger.info("STT reconnected — resuming")

                    elif action == "connection_lost":
                        from aether.transport.core_msg import CoreMsg

                        msg = CoreMsg.text(
                            text="Connection lost. Please refresh.",
                            user_id=user_id,
                            session_id=session_id,
                            role="system",
                            transport="status",
                        )
                        await callback(msg)
                        logger.error("STT connection permanently lost")

        except asyncio.CancelledError:
            logger.info("STT event handler cancelled")
        except Exception as e:
            logger.error(f"STT event handler error: {e}", exc_info=True)

    async def _debounce_and_trigger(
        self,
        user_id: str,
        session_id: str,
        session_state: dict[str, Any],
        callback: Any,
    ) -> None:
        """Wait for silence, then trigger the voice response."""
        try:
            await asyncio.sleep(config_module.config.server.debounce_delay)

            transcript = session_state.get("accumulated_transcript", "").strip()
            session_state["accumulated_transcript"] = ""

            if not transcript:
                return

            await self._trigger_voice_response(
                user_id, session_id, session_state, transcript, callback
            )

        except asyncio.CancelledError:
            logger.debug("Debounce reset (user still speaking)")
        except Exception as e:
            logger.error(f"Debounce/trigger error: {e}", exc_info=True)

    async def _trigger_voice_response(
        self,
        user_id: str,
        session_id: str,
        session_state: dict[str, Any],
        transcript: str,
        callback: Any,
    ) -> None:
        """Run the voice pipeline for a complete utterance from STT streaming."""
        from aether.transport.core_msg import CoreMsg

        session_state["is_responding"] = True

        # Send "thinking..." status
        await callback(
            CoreMsg.text(
                text="thinking...",
                user_id=user_id,
                session_id=session_id,
                role="system",
                transport="status",
            )
        )

        try:
            session_state.setdefault("turn_count", 0)
            session_state["turn_count"] += 1
            logger.info(f'STT: "{transcript}"')

            # Send final transcript
            await callback(
                CoreMsg.text(
                    text=transcript,
                    user_id=user_id,
                    session_id=session_id,
                    role="system",
                    transport="transcript",
                )
            )

            # Run voice pipeline — send each response via callback
            async for resp in self._run_voice_pipeline(
                transcript, user_id, session_id, session_state
            ):
                await callback(resp)

        except Exception as e:
            logger.error(f"Response pipeline error: {e}", exc_info=True)
        finally:
            session_state["is_responding"] = False
            await callback(
                CoreMsg.text(
                    text="listening...",
                    user_id=user_id,
                    session_id=session_id,
                    role="system",
                    transport="status",
                )
            )

    # ─── Session Greeting ────────────────────────────────────────

    async def _session_greeting(
        self,
        user_id: str,
        session_id: str,
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """Generate and return a personalized greeting with TTS."""
        from aether.greeting import generate_greeting
        from aether.transport.core_msg import CoreMsg

        session_state["is_responding"] = True
        tts = self._get_tts(session_id)

        try:
            greeting_text = await generate_greeting(
                self._memory_store, self._llm_provider
            )

            yield CoreMsg.text(
                text=greeting_text,
                user_id=user_id,
                session_id=session_id,
                role="assistant",
                transport="text_chunk",
            )

            # TTS
            if tts:
                async for audio_msg in self._synthesize(
                    greeting_text, user_id, session_id, tts
                ):
                    yield audio_msg

            yield CoreMsg.event(
                event_type="stream_end",
                user_id=user_id,
                session_id=session_id,
                transport="control",
            )

        except Exception as e:
            logger.error(f"Greeting error: {e}", exc_info=True)
        finally:
            session_state["is_responding"] = False

    # ─── Helpers ─────────────────────────────────────────────────

    def _get_tts(self, session_id: str) -> "TTSProcessor | None":
        """Get TTS processor for a session."""
        return self._tts_processors.get(session_id)

    async def _synthesize(
        self,
        text: str,
        user_id: str,
        session_id: str,
        tts: "TTSProcessor",
        sentence_index: int = 0,
    ) -> AsyncGenerator["CoreMsg", None]:
        """Synthesize text to audio and yield audio CoreMsg."""
        from aether.transport.core_msg import CoreMsg

        try:
            frame = text_frame(text, role="assistant")
            async for tts_out in tts.process(frame):
                if tts_out.type == FrameType.AUDIO:
                    yield CoreMsg.audio(
                        audio_data=tts_out.data,
                        user_id=user_id,
                        session_id=session_id,
                        sample_rate=tts_out.metadata.get("sample_rate", 24000),
                        transport="audio_chunk",
                        sentence_index=sentence_index,
                    )
        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def _retrieve_memory(self, user_text: str) -> str | None:
        """Retrieve relevant memories for context injection."""
        try:
            from aether.core.config import config

            results = await self._memory_store.search(
                user_text, limit=config.memory.search_limit
            )
            if not results:
                return None

            lines = []
            for r in results:
                if r.get("type") == "fact":
                    lines.append(f"[Known fact] {r['fact']}")
                elif r.get("type") == "action":
                    output_preview = r.get("output", "")[:100]
                    status = "failed" if r.get("error") else "succeeded"
                    lines.append(
                        f"[Past action] Used {r['tool_name']}({r.get('arguments', '{}')}) — "
                        f"{status}: {output_preview}"
                    )
                elif r.get("type") == "session":
                    lines.append(f"[Previous session] {r['summary']}")
                elif r.get("type") == "conversation":
                    lines.append(
                        f"[Previous conversation] User said: {r['user_message']} — "
                        f"You replied: {r['assistant_message']}"
                    )

            return "\n".join(lines) if lines else None

        except Exception as e:
            logger.error(f"Memory retrieval error: {e}", exc_info=True)
            return None
