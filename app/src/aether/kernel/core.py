"""
Kernel Core — the single CoreInterface implementation.

Replaces CoreHandler as the adapter between the transport layer (CoreMsg)
and the Aether core (providers, memory, tools, skills, plugins).

Architecture:
  Transport → TransportManager → KernelCore.process_message()
                                    ├─ TextContent  → scheduler (reply job)
                                    ├─ AudioContent → STT → scheduler (reply job)
                                    └─ EventContent → event handling

All text/audio replies are submitted to the KernelScheduler which routes
through the ServiceRouter to ReplyService, MemoryService, etc.
KernelCore owns the CoreInterface contract.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any, AsyncGenerator

import aether.core.config as config_module
from aether.core.frames import Frame, FrameType, text_frame, vision_frame
from aether.greeting import generate_greeting
from aether.kernel.contracts import JobPriority, KernelRequest
from aether.memory.store import MemoryStore
from aether.plugins.context import PluginContextStore
from aether.processors.stt import STTProcessor
from aether.processors.tts import TTSProcessor
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


class KernelCore(CoreInterface):
    """
    The kernel — single CoreInterface implementation for all transports.

    Handles:
    - Provider lifecycle (start/stop STT, LLM, TTS, Memory)
    - Per-session state (conversation history, STT streaming, TTS, vision)
    - Scheduler submission: text/audio → KernelScheduler → ServiceRouter
    - TTS synthesis for voice mode (inline on scheduler stream)
    - STT streaming with debounce + utterance detection
    - Session greeting on voice connect
    - Session summary on disconnect
    - Event handling (mute, unmute, image, config, notification feedback)
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

        # Callback for background messages (STT interim, status audio, etc.)
        self._background_callback: Any = None

        # Scheduler — created in start()
        self._scheduler: Any = None

    # ─── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start all providers and the scheduler."""
        await self.stt_provider.start()
        await self.llm_provider.start()
        await self.tts_provider.start()
        await self.memory_store.start()

        # Build the service layer
        from aether.kernel.scheduler import KernelScheduler, ServiceRouter
        from aether.llm.context_builder import ContextBuilder
        from aether.llm.core import LLMCore
        from aether.services import (
            MemoryService,
            NotificationService,
            ReplyService,
            ToolService,
        )
        from aether.tools.orchestrator import ToolOrchestrator

        tool_orchestrator = ToolOrchestrator(self.tool_registry)
        llm_core = LLMCore(self.llm_provider, tool_orchestrator)
        context_builder = ContextBuilder(
            skill_loader=self.skill_loader,
            plugin_context_store=self.plugin_context,
            tool_registry=self.tool_registry,
            memory_store=self.memory_store,
        )

        reply_service = ReplyService(llm_core, context_builder, self.memory_store)
        memory_service = MemoryService(llm_core, self.memory_store)
        notification_service = NotificationService(llm_core, self.memory_store)
        tool_service = ToolService(tool_orchestrator)

        router = ServiceRouter(
            reply_service=reply_service,
            memory_service=memory_service,
            notification_service=notification_service,
            tool_service=tool_service,
        )

        cfg = config_module.config.kernel
        self._scheduler = KernelScheduler(
            service_router=router,
            max_interactive_workers=cfg.workers_interactive,
            max_background_workers=cfg.workers_background,
            interactive_queue_limit=cfg.interactive_queue_limit,
            background_queue_limit=cfg.background_queue_limit,
        )
        await self._scheduler.start()
        logger.info(
            "Kernel started: %d P-Cores, %d E-Cores",
            cfg.workers_interactive,
            cfg.workers_background,
        )

        logger.info("Kernel core started")

    async def stop(self) -> None:
        """Stop all sessions, scheduler, and providers."""
        for session_id in list(self._sessions):
            await self._cleanup_session(session_id)

        if self._scheduler is not None:
            await self._scheduler.stop()
            self._scheduler = None

        await self.stt_provider.stop()
        await self.llm_provider.stop()
        await self.tts_provider.stop()
        await self.memory_store.stop()
        logger.info("Kernel core stopped")

    async def health_check(self) -> dict:
        """Check health of all components."""
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

    # ─── Background Callback ─────────────────────────────────────

    def set_status_audio_callback(self, callback) -> None:
        """Set callback for background messages (STT interim, status audio, etc.)."""
        self._background_callback = callback

    # ─── Session Management ──────────────────────────────────────

    def _get_session(self, session_id: str, mode: str = "voice") -> "_SessionState":
        """Get or create session state."""
        if session_id not in self._sessions:
            session = _SessionState(
                session_id=session_id,
                mode=mode,
                started_at=time.time(),
            )

            if mode == "voice":
                session.stt = STTProcessor(self.stt_provider)
                session.tts = TTSProcessor(self.tts_provider)

            self._sessions[session_id] = session

        return self._sessions[session_id]

    async def _ensure_session_started(self, session: "_SessionState") -> None:
        """Start session processors if not already started."""
        if session.started:
            return

        if session.stt:
            await session.stt.start()
        if session.tts:
            await session.tts.start()
        session.started = True

    async def _ensure_voice_processors(self, session: "_SessionState") -> None:
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

        if session.debounce_task and not session.debounce_task.done():
            session.debounce_task.cancel()

        if session.stt_event_task and not session.stt_event_task.done():
            session.stt_event_task.cancel()
            try:
                await session.stt_event_task
            except asyncio.CancelledError:
                pass

        await self.stt_provider.disconnect_stream()

        if session.turn_count > 0:
            await self._summarize_session(session)

        if session.stt:
            await session.stt.stop()
        if session.tts:
            await session.tts.stop()

        logger.info(f"Session {session_id} cleaned up (turns={session.turn_count})")

    # ─── Main Message Processing ─────────────────────────────────

    async def process_message(self, msg: CoreMsg) -> AsyncGenerator[CoreMsg, None]:
        """
        Process a message through the kernel pipeline.

        Routes based on content type:
        - TextContent → scheduler reply job (with inline TTS for voice)
        - AudioContent → batch STT → scheduler reply job
        - EventContent → event handling (stream_start, mute, config, etc.)
        """
        session_id = msg.session_id
        mode = msg.metadata.session_mode
        session = self._get_session(session_id, mode)
        await self._ensure_session_started(session)

        content = msg.content

        if isinstance(content, TextContent):
            async for resp in self._handle_text_via_scheduler(msg, session):
                yield resp

        elif isinstance(content, AudioContent):
            async for resp in self._handle_audio(msg, session):
                yield resp

        elif isinstance(content, EventContent):
            async for resp in self._handle_event(msg, session):
                yield resp

    # ─── Scheduler Path ──────────────────────────────────────────

    async def _handle_text_via_scheduler(
        self, msg: CoreMsg, session: "_SessionState"
    ) -> AsyncGenerator[CoreMsg, None]:
        """Submit a text reply job to the scheduler and stream results back."""
        content: TextContent = msg.content  # type: ignore[assignment]
        user_text = content.text
        session.turn_count += 1

        # Build vision payload from pending session state
        vision_payload: dict | None = None
        if session.pending_vision is not None:
            vision_payload = {
                "data": session.pending_vision.data,
                "mime_type": session.pending_vision.metadata.get(
                    "mime_type", "image/jpeg"
                ),
            }
            session.pending_vision = None

        kind = "reply_voice" if session.mode == "voice" else "reply_text"
        request = KernelRequest(
            kind=kind,
            modality=session.mode,
            user_id=msg.user_id,
            session_id=session.session_id,
            payload={
                "text": user_text,
                "history": list(session.conversation_history),
                "pending_vision": vision_payload,
            },
            priority=JobPriority.INTERACTIVE.value,
        )

        job_id = await self._scheduler.submit(request)

        # Ensure voice processors are ready for TTS
        is_voice = session.mode == "voice"
        if is_voice:
            await self._ensure_voice_processors(session)

        # Stream results and collect assistant text for history
        collected_text: list[str] = []
        sentence_index = 0

        async for event in self._scheduler.stream(job_id):
            if event.stream_type == "text_chunk":
                chunk_text = event.payload.get("text", "")
                collected_text.append(chunk_text)

                # Yield text chunk
                yield CoreMsg.text(
                    text=chunk_text,
                    user_id=msg.user_id,
                    session_id=session.session_id,
                    role="assistant",
                    transport="text_chunk",
                    sentence_index=sentence_index,
                )

                # Synthesize TTS for voice mode
                if is_voice and session.tts and chunk_text.strip():
                    try:
                        tts_frame = text_frame(chunk_text, role="assistant")
                        async for tts_out in session.tts.process(tts_frame):
                            if tts_out.type == FrameType.AUDIO:
                                yield CoreMsg.audio(
                                    audio_data=tts_out.data,
                                    user_id=msg.user_id,
                                    session_id=session.session_id,
                                    sample_rate=tts_out.metadata.get(
                                        "sample_rate", 24000
                                    ),
                                    transport="audio_chunk",
                                    sentence_index=sentence_index,
                                )
                    except Exception as e:
                        logger.error(f"TTS error for sentence {sentence_index}: {e}")

                sentence_index += 1

            elif event.stream_type == "status":
                status_text = event.payload.get("message", "Working...")
                yield CoreMsg.text(
                    text=status_text,
                    user_id=msg.user_id,
                    session_id=session.session_id,
                    role="system",
                    transport="status",
                )
                # Fire-and-forget TTS for status in voice mode
                if is_voice and session.tts:
                    asyncio.create_task(
                        self._speak_status(
                            msg.user_id, session.session_id, session.tts, status_text
                        )
                    )

            elif event.stream_type == "tool_result":
                yield CoreMsg.event(
                    event_type="tool_result",
                    user_id=msg.user_id,
                    session_id=session.session_id,
                    payload={
                        "name": event.payload.get("tool_name", ""),
                        "output": event.payload.get("output", "")[:500],
                        "error": event.payload.get("error", False),
                    },
                    transport="tools",
                )

            elif event.stream_type in ("stream_end", "done"):
                yield CoreMsg.event(
                    event_type="stream_end",
                    user_id=msg.user_id,
                    session_id=session.session_id,
                    transport="control",
                )

            elif event.stream_type == "error":
                yield CoreMsg.text(
                    text=f"[error] {event.payload.get('message', 'Unknown error')}",
                    user_id=msg.user_id,
                    session_id=session.session_id,
                    role="system",
                    transport="status",
                )

        # Update conversation history
        session.conversation_history.append({"role": "user", "content": user_text})
        assistant_text = " ".join(collected_text).strip()
        if assistant_text:
            session.conversation_history.append(
                {"role": "assistant", "content": assistant_text}
            )

    async def _handle_audio(
        self, msg: CoreMsg, session: "_SessionState"
    ) -> AsyncGenerator[CoreMsg, None]:
        """Handle batch audio — STT then route through scheduler."""
        content: AudioContent = msg.content  # type: ignore[assignment]
        await self._ensure_voice_processors(session)

        user_text = await self.stt_provider.transcribe(content.audio_data)

        if user_text:
            yield CoreMsg.text(
                text=user_text,
                user_id=msg.user_id,
                session_id=session.session_id,
                role="system",
                transport="transcript",
            )

            # Route through scheduler (same as text messages)
            synthetic_msg = CoreMsg.text(
                text=user_text,
                user_id=msg.user_id,
                session_id=session.session_id,
                role="user",
                transport="voice",
                session_mode=session.mode,
            )
            async for resp in self._handle_text_via_scheduler(synthetic_msg, session):
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
        self, msg: CoreMsg, session: "_SessionState"
    ) -> AsyncGenerator[CoreMsg, None]:
        """Handle event messages (stream_start, mute, config, etc.)."""
        content: EventContent = msg.content  # type: ignore[assignment]
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

    # ─── Helpers ─────────────────────────────────────────────────

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
                    if self._background_callback:
                        msg = CoreMsg.audio(
                            audio_data=tts_out.data,
                            user_id=user_id,
                            session_id=session_id,
                            transport="status_audio",
                        )
                        await self._background_callback(msg)
        except Exception as e:
            logger.debug(f"Status TTS skipped: {e}")

    # ─── STT Streaming Event Loop ────────────────────────────────

    async def _stt_event_loop(self, user_id: str, session: "_SessionState") -> None:
        """Background task: listen to streaming STT events."""
        try:
            async for event in self.stt_provider.stream_events():
                # Always drop everything when muted
                if session.is_muted:
                    continue

                # While responding, only drop interim transcripts (live captions).
                # utterance_end must still be processed so the debounce can fire
                # and the user's words are queued for the next turn.
                if (
                    session.is_responding
                    and event.type == FrameType.TEXT
                    and event.metadata.get("interim")
                ):
                    continue

                if event.type == FrameType.TEXT and event.metadata.get("interim"):
                    if self._background_callback:
                        msg = CoreMsg.text(
                            text=event.data,
                            user_id=user_id,
                            session_id=session.session_id,
                            role="system",
                            transport="transcript_interim",
                        )
                        await self._background_callback(msg)

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
                            if (
                                session.debounce_task
                                and not session.debounce_task.done()
                            ):
                                session.debounce_task.cancel()
                            session.debounce_task = asyncio.create_task(
                                self._debounce_and_trigger(user_id, session)
                            )

                    elif action == "reconnected":
                        if self._background_callback:
                            msg = CoreMsg.text(
                                text="listening...",
                                user_id=user_id,
                                session_id=session.session_id,
                                role="system",
                                transport="status",
                            )
                            await self._background_callback(msg)
                        logger.info("STT reconnected — resuming")

                    elif action == "connection_lost":
                        if self._background_callback:
                            msg = CoreMsg.text(
                                text="Connection lost. Please refresh.",
                                user_id=user_id,
                                session_id=session.session_id,
                                role="system",
                                transport="status",
                            )
                            await self._background_callback(msg)
                        logger.error("STT connection permanently lost")

        except asyncio.CancelledError:
            logger.info("STT event handler cancelled")
        except Exception as e:
            logger.error(f"STT event handler error: {e}", exc_info=True)

    async def _debounce_and_trigger(
        self, user_id: str, session: "_SessionState"
    ) -> None:
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
        self, user_id: str, session: "_SessionState", transcript: str
    ) -> None:
        """Run the voice pipeline for a complete utterance from STT streaming.

        Submits through the scheduler (same path as WebSocket text messages).
        Results are sent via _background_callback since this runs as a
        background task from the STT event loop.
        """
        session.is_responding = True

        if self._background_callback:
            await self._background_callback(
                CoreMsg.text(
                    text="thinking...",
                    user_id=user_id,
                    session_id=session.session_id,
                    role="system",
                    transport="status",
                )
            )

        try:
            logger.info(f'STT: "{transcript}"')

            if self._background_callback:
                await self._background_callback(
                    CoreMsg.text(
                        text=transcript,
                        user_id=user_id,
                        session_id=session.session_id,
                        role="system",
                        transport="transcript",
                    )
                )

            # Build a synthetic CoreMsg to route through the scheduler path
            synthetic_msg = CoreMsg.text(
                text=transcript,
                user_id=user_id,
                session_id=session.session_id,
                role="user",
                transport="voice",
                session_mode="voice",
            )

            async for resp in self._handle_text_via_scheduler(synthetic_msg, session):
                if self._background_callback:
                    await self._background_callback(resp)

        except Exception as e:
            logger.error(f"Response pipeline error: {e}", exc_info=True)
        finally:
            session.is_responding = False
            if self._background_callback:
                await self._background_callback(
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
        self, user_id: str, session: "_SessionState"
    ) -> AsyncGenerator[CoreMsg, None]:
        """Generate and return a personalized greeting."""
        session.is_responding = True
        sid = session.session_id

        try:
            greeting_text = await generate_greeting(
                self.memory_store, self.llm_provider
            )

            yield CoreMsg.text(
                text=greeting_text,
                user_id=user_id,
                session_id=sid,
                role="assistant",
                transport="text_chunk",
            )

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

    async def _summarize_session(self, session: "_SessionState") -> None:
        """Summarize a session and store for cross-session continuity."""
        try:
            conv_lines = []
            tools_used: set[str] = set()
            for msg in session.conversation_history[-20:]:
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
        "conversation_history",
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
        started_at: float,
    ):
        self.session_id = session_id
        self.mode = mode
        self.conversation_history: list[dict] = []
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
