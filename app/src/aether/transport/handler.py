"""
Core Handler - Implementation of CoreInterface using existing Aether components.

This is the adapter between the transport layer (CoreMsg) and the Aether Core
(Frames). It contains the full processing pipeline:
  - STT streaming with debounce + utterance detection
  - Memory retrieval
  - LLM agentic loop (with tool calling)
  - TTS synthesis (voice mode)
  - Session greeting on connect
  - Session summarization on disconnect
  - Vision accumulation
  - Notification feedback → memory
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any, AsyncGenerator

import aether.core.config as config_module
from aether.core.frames import Frame, FrameType, audio_frame, text_frame, vision_frame
from aether.core.logging import PipelineTimer
from aether.greeting import generate_greeting
from aether.memory.store import MemoryStore
from aether.plugins.context import PluginContextStore
from aether.processors.llm import LLMProcessor
from aether.processors.memory import MemoryRetrieverProcessor
from aether.processors.stt import STTProcessor
from aether.processors.tts import TTSProcessor
from aether.processors.vision import VisionProcessor
from aether.providers.base import LLMProvider, STTProvider, TTSProvider
from aether.skills.loader import SkillLoader
from aether.tools.registry import ToolRegistry
from aether.transport.core_msg import (
    AudioContent,
    CoreMsg,
    EventContent,
    MsgDirection,
    TextContent,
)
from aether.transport.interface import CoreInterface

logger = logging.getLogger(__name__)

SESSION_SUMMARY_PROMPT = """Summarize this conversation session in 2-3 sentences.
Focus on what was accomplished (files created, questions answered, tasks completed), not what was said.
If tools were used, mention what they did.

Conversation:
{conversation}

Summary:"""


class CoreHandler(CoreInterface):
    """
    Implements CoreInterface using existing Aether components.

    This is the adapter that translates between CoreMsg (transport layer format)
    and Frames (Aether Core format). It contains the full pipeline logic that
    was previously in main.py's websocket_endpoint.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        memory_store: MemoryStore,
        tool_registry: ToolRegistry,
        skill_loader: SkillLoader,
        plugin_context: PluginContextStore,
        stt_provider: STTProvider,
        tts_provider: TTSProvider,
    ):
        self.llm_provider = llm_provider
        self.memory_store = memory_store
        self.tool_registry = tool_registry
        self.skill_loader = skill_loader
        self.plugin_context = plugin_context
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider

        # Per-session state
        self._sessions: dict[str, _SessionState] = {}

        # Callback for status audio — set by TransportManager
        self._status_audio_callback: Any = None

    # ─── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the core — initialize providers."""
        await self.stt_provider.start()
        await self.llm_provider.start()
        await self.tts_provider.start()
        await self.memory_store.start()
        logger.info("Core handler started")

    async def stop(self) -> None:
        """Stop the core — cleanup all sessions and providers."""
        for session_id in list(self._sessions):
            await self._cleanup_session(session_id)

        await self.stt_provider.stop()
        await self.llm_provider.stop()
        await self.tts_provider.stop()
        await self.memory_store.stop()
        logger.info("Core handler stopped")

    async def health_check(self) -> dict:
        """Check health of all core components."""
        stt_health = await self.stt_provider.health_check()
        llm_health = await self.llm_provider.health_check()
        tts_health = await self.tts_provider.health_check()
        facts = await self.memory_store.get_facts()

        return {
            "status": "ok",
            "providers": {
                "stt": stt_health,
                "llm": llm_health,
                "tts": tts_health,
            },
            "memory": {"facts_count": len(facts)},
            "tools": self.tool_registry.tool_names() if self.tool_registry else [],
            "skills": (
                [s.name for s in self.skill_loader.all()] if self.skill_loader else []
            ),
            "sessions": len(self._sessions),
        }

    # ─── Session Management ──────────────────────────────────────

    def _get_session(self, session_id: str, mode: str = "voice") -> _SessionState:
        """Get or create session state."""
        if session_id not in self._sessions:
            memory_retriever = MemoryRetrieverProcessor(self.memory_store)
            llm = LLMProcessor(
                self.llm_provider,
                self.memory_store,
                tool_registry=self.tool_registry,
                skill_loader=self.skill_loader,
                plugin_context=self.plugin_context,
            )
            llm.session_id = session_id

            session = _SessionState(
                session_id=session_id,
                mode=mode,
                memory_retriever=memory_retriever,
                llm=llm,
                started_at=time.time(),
            )

            # Create STT/TTS for voice mode
            if mode == "voice":
                session.stt = STTProcessor(self.stt_provider)
                session.tts = TTSProcessor(self.tts_provider)

            self._sessions[session_id] = session

        return self._sessions[session_id]

    async def _ensure_session_started(self, session: _SessionState) -> None:
        """Start session processors if not already started."""
        if session.started:
            return

        await session.llm.start()
        if session.stt:
            await session.stt.start()
        if session.tts:
            await session.tts.start()
        session.started = True

    async def _ensure_voice_processors(self, session: _SessionState) -> None:
        """Lazily create and start voice processors if not present."""
        if session.stt is None:
            session.stt = STTProcessor(self.stt_provider)
            await session.stt.start()
        if session.tts is None:
            session.tts = TTSProcessor(self.tts_provider)
            await session.tts.start()

    async def _cleanup_session(self, session_id: str) -> None:
        """Clean up a session — summarize and stop processors."""
        session = self._sessions.pop(session_id, None)
        if not session:
            return

        # Cancel any pending debounce
        if session.debounce_task and not session.debounce_task.done():
            session.debounce_task.cancel()

        # Cancel STT event listener
        if session.stt_event_task and not session.stt_event_task.done():
            session.stt_event_task.cancel()
            try:
                await session.stt_event_task
            except asyncio.CancelledError:
                pass

        await self.stt_provider.disconnect_stream()

        # Session summary before stopping LLM
        if session.turn_count > 0:
            await self._summarize_session(session)

        # Stop processors
        if session.stt:
            await session.stt.stop()
        await session.llm.stop()
        if session.tts:
            await session.tts.stop()

        logger.info(f"Session {session_id} cleaned up (turns={session.turn_count})")

    # ─── Main Message Processing ─────────────────────────────────

    async def process_message(self, msg: CoreMsg) -> AsyncGenerator[CoreMsg, None]:
        """
        Process a message through the core pipeline.

        Routes based on content type:
        - TextContent → text pipeline (Memory → LLM → optional TTS)
        - AudioContent → batch STT → text pipeline
        - EventContent → event handling (stream_start, mute, config, etc.)
        """
        session_id = msg.session_id
        mode = msg.metadata.session_mode
        session = self._get_session(session_id, mode)
        await self._ensure_session_started(session)

        content = msg.content

        if isinstance(content, TextContent):
            async for resp in self._handle_text(msg, session):
                yield resp

        elif isinstance(content, AudioContent):
            async for resp in self._handle_audio(msg, session):
                yield resp

        elif isinstance(content, EventContent):
            async for resp in self._handle_event(msg, session):
                yield resp

    # ─── Text Handling ───────────────────────────────────────────

    async def _handle_text(
        self, msg: CoreMsg, session: _SessionState
    ) -> AsyncGenerator[CoreMsg, None]:
        """Handle text message — run through LLM pipeline."""
        content: TextContent = msg.content
        user_text = content.text
        session.turn_count += 1

        if session.mode == "text":
            async for resp in self._run_text_pipeline(user_text, msg.user_id, session):
                yield resp
        else:
            await self._ensure_voice_processors(session)
            async for resp in self._run_voice_pipeline(user_text, msg.user_id, session):
                yield resp

    async def _handle_audio(
        self, msg: CoreMsg, session: _SessionState
    ) -> AsyncGenerator[CoreMsg, None]:
        """Handle batch audio — STT then LLM pipeline."""
        content: AudioContent = msg.content
        await self._ensure_voice_processors(session)

        # Batch transcribe
        user_text = await self.stt_provider.transcribe(content.audio_data)

        if user_text:
            # Send transcript
            yield CoreMsg.text(
                text=user_text,
                user_id=msg.user_id,
                session_id=session.session_id,
                role="system",
                transport="transcript",
            )

            session.turn_count += 1
            async for resp in self._run_voice_pipeline(user_text, msg.user_id, session):
                yield resp
        else:
            yield CoreMsg.text(
                text="Didn't catch that, try again",
                user_id=msg.user_id,
                session_id=session.session_id,
                role="system",
                transport="status",
            )

    # ─── Event Handling ──────────────────────────────────────────

    async def _handle_event(
        self, msg: CoreMsg, session: _SessionState
    ) -> AsyncGenerator[CoreMsg, None]:
        """Handle event messages (stream_start, mute, config, etc.)."""
        content: EventContent = msg.content
        event_type = content.event_type

        if event_type == "session_config":
            new_mode = content.payload.get("mode", "voice")
            if new_mode in ("text", "voice"):
                session.mode = new_mode
                if new_mode == "voice":
                    await self._ensure_voice_processors(session)
                logger.info(f"Session {session.session_id} mode: {new_mode}")
            yield CoreMsg.text(
                text="connected",
                user_id=msg.user_id,
                session_id=session.session_id,
                role="system",
                transport="status",
            )

        elif event_type == "stream_start":
            await self._ensure_voice_processors(session)
            session.mode = "voice"
            await self.stt_provider.connect_stream()

            # Start STT event listener
            session.stt_event_task = asyncio.create_task(
                self._stt_event_loop(msg.user_id, session)
            )

            yield CoreMsg.text(
                text="listening...",
                user_id=msg.user_id,
                session_id=session.session_id,
                role="system",
                transport="status",
            )

            # Session greeting
            is_reconnect = content.payload.get("reconnect", False)
            if not is_reconnect:
                async for resp in self._session_greeting(msg.user_id, session):
                    yield resp

        elif event_type == "stream_stop":
            if session.stt_event_task:
                session.stt_event_task.cancel()
                try:
                    await session.stt_event_task
                except asyncio.CancelledError:
                    pass
                session.stt_event_task = None
            await self.stt_provider.disconnect_stream()

        elif event_type == "audio_chunk":
            # Streaming audio chunk → forward to STT provider
            audio_b64 = content.payload.get("data", "")
            if audio_b64 and not session.is_muted:
                audio_data = base64.b64decode(audio_b64)
                await self.stt_provider.send_audio(audio_data)

        elif event_type == "mute":
            session.is_muted = True
            if session.debounce_task and not session.debounce_task.done():
                session.debounce_task.cancel()
            session.accumulated_transcript = ""
            yield CoreMsg.text(
                text="muted",
                user_id=msg.user_id,
                session_id=session.session_id,
                role="system",
                transport="status",
            )

        elif event_type == "unmute":
            session.is_muted = False
            yield CoreMsg.text(
                text="listening...",
                user_id=msg.user_id,
                session_id=session.session_id,
                role="system",
                transport="status",
            )

        elif event_type == "image":
            image_b64 = content.payload.get("data", "")
            mime = content.payload.get("mime", "image/jpeg")
            if image_b64:
                image_data = base64.b64decode(image_b64)
                session.pending_vision = vision_frame(image_data, mime_type=mime)
            yield CoreMsg.text(
                text="Image received, listening...",
                user_id=msg.user_id,
                session_id=session.session_id,
                role="system",
                transport="status",
            )

        elif event_type == "notification_feedback":
            await self._handle_notification_feedback(content.payload)

        elif event_type == "disconnect":
            await self._cleanup_session(session.session_id)

    # ─── Text Pipeline (no TTS) ─────────────────────────────────

    async def _run_text_pipeline(
        self, user_text: str, user_id: str, session: _SessionState
    ) -> AsyncGenerator[CoreMsg, None]:
        """Memory → LLM → text_chunk. No TTS."""
        timer = PipelineTimer()
        sid = session.session_id

        # Memory retrieval
        pre_frames = await self._gather_pre_frames(user_text, session)
        timer.mark("Memory")

        # LLM streaming
        sentence_index = 0
        llm_marked = False

        for pf in pre_frames:
            async for llm_frame in session.llm.process(pf):
                if (
                    llm_frame.type == FrameType.TEXT
                    and llm_frame.metadata.get("role") == "assistant"
                ):
                    if not llm_marked:
                        timer.mark("LLM")
                        llm_marked = True

                    yield CoreMsg.text(
                        text=llm_frame.data,
                        user_id=user_id,
                        session_id=sid,
                        role="assistant",
                        transport="text_chunk",
                        sentence_index=sentence_index,
                    )
                    sentence_index += 1

                elif llm_frame.type == FrameType.STATUS:
                    yield CoreMsg.text(
                        text=llm_frame.data.get("text", "Working..."),
                        user_id=user_id,
                        session_id=sid,
                        role="system",
                        transport="status",
                    )

                elif llm_frame.type == FrameType.TOOL_RESULT:
                    yield CoreMsg.event(
                        event_type="tool_result",
                        user_id=user_id,
                        session_id=sid,
                        payload={
                            "name": llm_frame.data["tool_name"],
                            "output": llm_frame.data["output"][:500],
                            "error": llm_frame.data.get("error", False),
                        },
                        transport="tools",
                    )

                elif llm_frame.type == FrameType.CONTROL:
                    if llm_frame.data.get("action") == "llm_done":
                        yield CoreMsg.event(
                            event_type="stream_end",
                            user_id=user_id,
                            session_id=sid,
                            transport="control",
                        )

        timer.mark("Done")
        logger.info(f"[text] {sentence_index} sentences | {timer.summary()}")

    # ─── Voice Pipeline (with TTS) ──────────────────────────────

    async def _run_voice_pipeline(
        self, user_text: str, user_id: str, session: _SessionState
    ) -> AsyncGenerator[CoreMsg, None]:
        """Memory → LLM → TTS streaming pipeline."""
        timer = PipelineTimer()
        sid = session.session_id
        tts = session.tts
        assert tts is not None

        # Memory retrieval
        pre_frames = await self._gather_pre_frames(user_text, session)
        timer.mark("Memory")

        # LLM + TTS streaming
        sentence_index = 0
        total_audio_bytes = 0
        llm_marked = False

        for pf in pre_frames:
            async for llm_frame in session.llm.process(pf):
                if (
                    llm_frame.type == FrameType.TEXT
                    and llm_frame.metadata.get("role") == "assistant"
                ):
                    if not llm_marked:
                        timer.mark("LLM")
                        llm_marked = True

                    sentence_text = llm_frame.data

                    # Text chunk
                    yield CoreMsg.text(
                        text=sentence_text,
                        user_id=user_id,
                        session_id=sid,
                        role="assistant",
                        transport="text_chunk",
                        sentence_index=sentence_index,
                    )

                    # TTS
                    try:
                        async for tts_frame in tts.process(llm_frame):
                            if tts_frame.type == FrameType.AUDIO:
                                total_audio_bytes += len(tts_frame.data)
                                yield CoreMsg.audio(
                                    audio_data=tts_frame.data,
                                    user_id=user_id,
                                    session_id=sid,
                                    sample_rate=tts_frame.metadata.get(
                                        "sample_rate", 24000
                                    ),
                                    transport="audio_chunk",
                                    sentence_index=sentence_index,
                                )
                    except Exception as e:
                        logger.error(f"TTS error for sentence {sentence_index}: {e}")

                    sentence_index += 1

                elif llm_frame.type == FrameType.STATUS:
                    status_text = llm_frame.data.get("text", "Working...")
                    yield CoreMsg.text(
                        text=status_text,
                        user_id=user_id,
                        session_id=sid,
                        role="system",
                        transport="status",
                    )

                    # Fire-and-forget status TTS
                    asyncio.create_task(
                        self._speak_status(user_id, sid, tts, status_text)
                    )

                elif llm_frame.type == FrameType.TOOL_RESULT:
                    yield CoreMsg.event(
                        event_type="tool_result",
                        user_id=user_id,
                        session_id=sid,
                        payload={
                            "name": llm_frame.data["tool_name"],
                            "output": llm_frame.data["output"][:500],
                            "error": llm_frame.data.get("error", False),
                        },
                        transport="tools",
                    )

                elif llm_frame.type == FrameType.CONTROL:
                    if llm_frame.data.get("action") == "llm_done":
                        yield CoreMsg.event(
                            event_type="stream_end",
                            user_id=user_id,
                            session_id=sid,
                            transport="control",
                        )

        timer.mark("TTS")
        audio_kb = total_audio_bytes / 1024
        logger.info(
            f"LLM: {sentence_index} sentences | TTS: {audio_kb:.0f}KB | {timer.summary()}"
        )

    # ─── Helpers ─────────────────────────────────────────────────

    async def _gather_pre_frames(
        self, user_text: str, session: _SessionState
    ) -> list[Frame]:
        """Gather memory + vision frames before LLM processing."""
        pre_frames: list[Frame] = []
        user_frame = text_frame(user_text, role="user")

        # Vision (if pending)
        if session.pending_vision:
            async for f in VisionProcessor().process(session.pending_vision):
                pre_frames.append(f)
            session.pending_vision = None

        # Memory retrieval
        async for f in session.memory_retriever.process(user_frame):
            pre_frames.append(f)

        return pre_frames

    async def _speak_status(
        self,
        user_id: str,
        session_id: str,
        tts: TTSProcessor,
        text: str,
    ) -> None:
        """Fire-and-forget: synthesize status text as a short TTS clip."""
        try:
            status_f = text_frame(text, role="assistant")
            async for tts_out in tts.process(status_f):
                if tts_out.type == FrameType.AUDIO:
                    # This will be picked up by the _status_audio_callback if set
                    if self._status_audio_callback:
                        msg = CoreMsg.audio(
                            audio_data=tts_out.data,
                            user_id=user_id,
                            session_id=session_id,
                            transport="status_audio",
                        )
                        await self._status_audio_callback(msg)
        except Exception as e:
            logger.debug(f"Status TTS skipped: {e}")

    def set_status_audio_callback(self, callback) -> None:
        """Set callback for fire-and-forget status audio messages."""
        self._status_audio_callback = callback

    # ─── STT Streaming Event Loop ────────────────────────────────

    async def _stt_event_loop(self, user_id: str, session: _SessionState) -> None:
        """
        Background task: listen to streaming STT events.
        Half-duplex: drops events while assistant is responding or muted.
        """
        try:
            async for event in self.stt_provider.stream_events():
                if session.is_responding or session.is_muted:
                    continue

                if event.type == FrameType.TEXT and event.metadata.get("interim"):
                    # Interim transcript — forward to client
                    if self._status_audio_callback:
                        msg = CoreMsg.text(
                            text=event.data,
                            user_id=user_id,
                            session_id=session.session_id,
                            role="system",
                            transport="transcript_interim",
                        )
                        await self._status_audio_callback(msg)

                elif event.type == FrameType.CONTROL:
                    action = event.data.get("action")

                    if action == "utterance_end":
                        transcript = event.data.get("transcript", "")
                        if transcript:
                            session.accumulated_transcript += (
                                " " + transcript
                                if session.accumulated_transcript
                                else transcript
                            )
                            # Reset debounce
                            if (
                                session.debounce_task
                                and not session.debounce_task.done()
                            ):
                                session.debounce_task.cancel()
                            session.debounce_task = asyncio.create_task(
                                self._debounce_and_trigger(user_id, session)
                            )

                    elif action == "reconnected":
                        if self._status_audio_callback:
                            msg = CoreMsg.text(
                                text="listening...",
                                user_id=user_id,
                                session_id=session.session_id,
                                role="system",
                                transport="status",
                            )
                            await self._status_audio_callback(msg)
                        logger.info("STT reconnected — resuming")

                    elif action == "connection_lost":
                        if self._status_audio_callback:
                            msg = CoreMsg.text(
                                text="Connection lost. Please refresh.",
                                user_id=user_id,
                                session_id=session.session_id,
                                role="system",
                                transport="status",
                            )
                            await self._status_audio_callback(msg)
                        logger.error("STT connection permanently lost")

        except asyncio.CancelledError:
            logger.info("STT event handler cancelled")
        except Exception as e:
            logger.error(f"STT event handler error: {e}", exc_info=True)

    async def _debounce_and_trigger(self, user_id: str, session: _SessionState) -> None:
        """Wait for silence, then trigger the response."""
        try:
            await asyncio.sleep(config_module.config.server.debounce_delay)

            transcript = session.accumulated_transcript.strip()
            session.accumulated_transcript = ""

            if not transcript:
                return

            await self._trigger_voice_response(user_id, session, transcript)

        except asyncio.CancelledError:
            logger.debug("Debounce reset (user still speaking)")
        except Exception as e:
            logger.error(f"Debounce/trigger error: {e}", exc_info=True)

    async def _trigger_voice_response(
        self, user_id: str, session: _SessionState, transcript: str
    ) -> None:
        """Run the voice pipeline for a complete utterance from STT streaming."""
        session.is_responding = True

        # Send "thinking..." status
        if self._status_audio_callback:
            await self._status_audio_callback(
                CoreMsg.text(
                    text="thinking...",
                    user_id=user_id,
                    session_id=session.session_id,
                    role="system",
                    transport="status",
                )
            )

        try:
            session.turn_count += 1
            logger.info(f'STT: "{transcript}"')

            # Send final transcript
            if self._status_audio_callback:
                await self._status_audio_callback(
                    CoreMsg.text(
                        text=transcript,
                        user_id=user_id,
                        session_id=session.session_id,
                        role="system",
                        transport="transcript",
                    )
                )

            # Run voice pipeline — send each response via callback
            async for resp in self._run_voice_pipeline(transcript, user_id, session):
                if self._status_audio_callback:
                    await self._status_audio_callback(resp)

        except Exception as e:
            logger.error(f"Response pipeline error: {e}", exc_info=True)
        finally:
            session.is_responding = False
            if self._status_audio_callback:
                await self._status_audio_callback(
                    CoreMsg.text(
                        text="listening...",
                        user_id=user_id,
                        session_id=session.session_id,
                        role="system",
                        transport="status",
                    )
                )

    # ─── Session Greeting ────────────────────────────────────────

    async def _session_greeting(
        self, user_id: str, session: _SessionState
    ) -> AsyncGenerator[CoreMsg, None]:
        """Generate and return a personalized greeting."""
        session.is_responding = True
        sid = session.session_id

        try:
            greeting_text = await generate_greeting(
                self.memory_store, self.llm_provider
            )

            # Text chunk
            yield CoreMsg.text(
                text=greeting_text,
                user_id=user_id,
                session_id=sid,
                role="assistant",
                transport="text_chunk",
            )

            # TTS
            if session.tts:
                greeting_f = text_frame(greeting_text, role="assistant")
                try:
                    async for tts_out in session.tts.process(greeting_f):
                        if tts_out.type == FrameType.AUDIO:
                            yield CoreMsg.audio(
                                audio_data=tts_out.data,
                                user_id=user_id,
                                session_id=sid,
                                sample_rate=tts_out.metadata.get("sample_rate", 24000),
                                transport="audio_chunk",
                            )
                except Exception as e:
                    logger.error(f"Greeting TTS error: {e}")

            yield CoreMsg.event(
                event_type="stream_end",
                user_id=user_id,
                session_id=sid,
                transport="control",
            )

        except Exception as e:
            logger.error(f"Greeting error: {e}", exc_info=True)
        finally:
            session.is_responding = False

    # ─── Session Summary ─────────────────────────────────────────

    async def _summarize_session(self, session: _SessionState) -> None:
        """Summarize a session and store for cross-session continuity."""
        try:
            conv_lines = []
            tools_used = set()
            for msg in session.llm.conversation_history[-20:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    conv_lines.append(f"User: {content}")
                elif role == "assistant" and content:
                    conv_lines.append(f"Aether: {content}")
                elif role == "tool":
                    tools_used.add("tool")

            if not conv_lines:
                return

            conv_text = "\n".join(conv_lines)
            prompt = SESSION_SUMMARY_PROMPT.format(conversation=conv_text)

            summary = ""
            async for token in self.llm_provider.generate_stream(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3,
            ):
                summary += token

            summary = summary.strip()
            if not summary:
                return

            await self.memory_store.add_session_summary(
                session_id=session.session_id,
                summary=summary,
                started_at=session.started_at,
                ended_at=time.time(),
                turns=session.turn_count,
                tools_used=list(tools_used),
            )

        except Exception as e:
            logger.error(f"Session summary failed: {e}")

    # ─── Notification Feedback ───────────────────────────────────

    async def _handle_notification_feedback(self, data: dict) -> None:
        """Store notification feedback as a preference in memory."""
        feedback = data.get("action", "")
        plugin = data.get("plugin", "unknown")
        sender = data.get("sender", "")

        fact = ""
        if feedback == "engaged":
            fact = f"User immediately reads {plugin} notifications from {sender}"
        elif feedback == "dismissed":
            fact = f"User dismisses {plugin} notifications from {sender}"
        elif feedback == "muted":
            fact = f"User wants to mute all {plugin} notifications from {sender}"

        if fact:
            await self.memory_store.store_preference(fact)
            logger.info(f"Preference stored: {fact}")

    # ─── Notification Sending ────────────────────────────────────

    async def send_notification(self, user_id: str, notification: CoreMsg) -> None:
        """Send a notification — handled by TransportManager."""
        logger.info(f"Notification for {user_id}: {notification.content}")


class _SessionState:
    """Per-session state container."""

    __slots__ = (
        "session_id",
        "mode",
        "memory_retriever",
        "llm",
        "stt",
        "tts",
        "started",
        "started_at",
        "turn_count",
        "is_responding",
        "is_muted",
        "pending_vision",
        "accumulated_transcript",
        "debounce_task",
        "stt_event_task",
    )

    def __init__(
        self,
        session_id: str,
        mode: str,
        memory_retriever: MemoryRetrieverProcessor,
        llm: LLMProcessor,
        started_at: float,
    ):
        self.session_id = session_id
        self.mode = mode
        self.memory_retriever = memory_retriever
        self.llm = llm
        self.stt: STTProcessor | None = None
        self.tts: TTSProcessor | None = None
        self.started = False
        self.started_at = started_at
        self.turn_count = 0
        self.is_responding = False
        self.is_muted = False
        self.pending_vision: Frame | None = None
        self.accumulated_transcript = ""
        self.debounce_task: asyncio.Task | None = None
        self.stt_event_task: asyncio.Task | None = None
