"""
AgentCore — single facade for all transports.

Wraps the KernelScheduler behind simple async methods:
  - generate_reply()       → text chat (HTTP API)
  - generate_reply_voice() → voice chat (WebRTC)
  - generate_greeting()    → first-connection voice greeting
  - subscribe/broadcast    → notification sidecar (WS)

This replaces KernelCore (~920 lines) with a thin ~200-line facade.
All job scheduling, routing, and execution is handled by the scheduler
and services underneath — AgentCore just submits jobs and streams results.

Single-user, single-container: no auth, no multi-user isolation.

Session history is persisted to SQLite via SessionStore.
Task Ledger tracks all background work (memory extraction, nightly
analysis, sub-agent tasks) for audit and restart recovery.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable

from aether.core.metrics import metrics
from aether.kernel.contracts import (
    JobKind,
    JobModality,
    JobPriority,
    KernelEvent,
    KernelRequest,
)
from aether.session.models import TaskType

if TYPE_CHECKING:
    from aether.kernel.event_bus import EventBus
    from aether.kernel.contracts import KernelEvent
    from aether.kernel.scheduler import KernelScheduler
    from aether.llm.context_builder import ContextBuilder
    from aether.llm.core import LLMCore
    from aether.memory.store import MemoryStore
    from aether.plugins.context import PluginContextStore
    from aether.providers.base import LLMProvider
    from aether.services.nightly_analysis import NightlyAnalysisService
    from aether.session.ledger import TaskLedger
    from aether.session.store import SessionStore
    from aether.services.notification_service import NotificationDecision
    from aether.skills.loader import SkillLoader
    from aether.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentCore:
    """
    Single interface for all transports.

    Wraps the scheduler and services behind simple async methods.
    Transports (HTTP, WebRTC, WS) call AgentCore — they never touch
    the scheduler or services directly.
    """

    def __init__(
        self,
        scheduler: "KernelScheduler",
        memory_store: "MemoryStore",
        llm_provider: "LLMProvider",
        tool_registry: "ToolRegistry",
        skill_loader: "SkillLoader",
        plugin_context: "PluginContextStore",
        session_store: "SessionStore",
        task_ledger: "TaskLedger",
        event_bus: "EventBus | None" = None,
        llm_core: "LLMCore | None" = None,
        context_builder: "ContextBuilder | None" = None,
        nightly_service: "NightlyAnalysisService | None" = None,
    ) -> None:
        self._scheduler = scheduler
        self._memory_store = memory_store
        self._llm_provider = llm_provider
        self._tool_registry = tool_registry
        self._skill_loader = skill_loader
        self._plugin_context = plugin_context
        self._session_store = session_store
        self._task_ledger = task_ledger
        self._event_bus = event_bus
        self._llm_core = llm_core
        self._context_builder = context_builder
        self._nightly_service = nightly_service

        # Notification subscribers (WS sidecar connections)
        self._notification_subscribers: list[Callable] = []

        # Optional text-reply session handler (Gemini P-worker path)
        self._text_reply_handler: (
            Callable[
                [str, str, list[dict] | None, dict | None],
                AsyncGenerator["KernelEvent", None],
            ]
            | None
        ) = None

        # Voice transport reference for spoken notifications
        self._voice_transport: Any = None

        # Track last greeting time for time-gap logic
        self._last_greeting_at: float = 0.0

        # Briefing queue for temporal greetings (notifications/weather)
        self._briefing_items: list[dict[str, Any]] = []

        # Active session loop tasks (for background/autonomous work)
        self._session_tasks: dict[str, asyncio.Task] = {}

        # EventBus subscription task for task.completed events
        self._task_completed_listener: asyncio.Task | None = None
        self._task_completed_queue: asyncio.Queue | None = None

        # Background memory extraction handler (polls Task Ledger)
        self._memory_extraction_listener: asyncio.Task | None = None

        # Nightly analysis loop task (24-hour timer)
        self._nightly_analysis_task: asyncio.Task | None = None

        # Notification sweep task (60-second timer)
        self._notification_sweep_task: asyncio.Task | None = None

        # System session bootstrap flag for scheduled/background ledger tasks.
        self._system_session_ready = False

    # ─── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Start the scheduler (worker pools) and event subscriptions."""
        await self._scheduler.start()

        # Ensure the synthetic system session exists for scheduled jobs.
        await self._ensure_system_session()

        # Resume interrupted tasks from the Task Ledger (restart recovery)
        try:
            resumed = await self._task_ledger.resume_interrupted()
            if resumed > 0:
                logger.info("Resumed %d interrupted tasks from Task Ledger", resumed)
        except Exception as e:
            logger.warning("Failed to resume interrupted tasks: %s", e)

        # Subscribe to task.completed events from SubAgentManager
        if self._event_bus is not None:
            self._task_completed_queue = self._event_bus.subscribe("task.completed")
            self._task_completed_listener = asyncio.create_task(
                self._listen_task_completed()
            )

        # Start background memory extraction handler
        self._memory_extraction_listener = asyncio.create_task(
            self._process_memory_extractions()
        )

        # Start nightly analysis loop (24-hour timer)
        if self._nightly_service is not None:
            self._nightly_analysis_task = asyncio.create_task(
                self._run_nightly_analysis_loop()
            )

        # Start notification sweep (60-second timer)
        self._notification_sweep_task = asyncio.create_task(
            self._sweep_notification_queue()
        )

        logger.info("AgentCore started")

    async def stop(self) -> None:
        """Stop the scheduler and clean up."""
        # Cancel task.completed listener
        if self._task_completed_listener is not None:
            self._task_completed_listener.cancel()
            try:
                await self._task_completed_listener
            except asyncio.CancelledError:
                pass
            self._task_completed_listener = None

        if self._event_bus is not None and self._task_completed_queue is not None:
            self._event_bus.unsubscribe("task.completed", self._task_completed_queue)
            self._task_completed_queue = None

        # Cancel memory extraction handler
        if self._memory_extraction_listener is not None:
            self._memory_extraction_listener.cancel()
            try:
                await self._memory_extraction_listener
            except asyncio.CancelledError:
                pass
            self._memory_extraction_listener = None

        # Cancel nightly analysis loop
        if self._nightly_analysis_task is not None:
            self._nightly_analysis_task.cancel()
            try:
                await self._nightly_analysis_task
            except asyncio.CancelledError:
                pass
            self._nightly_analysis_task = None

        # Cancel notification sweep
        if self._notification_sweep_task is not None:
            self._notification_sweep_task.cancel()
            try:
                await self._notification_sweep_task
            except asyncio.CancelledError:
                pass
            self._notification_sweep_task = None

        await self._scheduler.stop()
        logger.info("AgentCore stopped")

    # ─── Reply (used by HTTP + Voice) ────────────────────────────

    async def generate_reply(
        self,
        text: str,
        session_id: str,
        history: list[dict] | None = None,
        vision: dict | None = None,
    ) -> AsyncGenerator[KernelEvent, None]:
        """
        Submit a text reply job to the scheduler. Yields KernelEvents.

        The caller (HTTP or VoiceSession) decides how to render them:
        - HTTP: convert to SSE / OpenAI-compatible chunks
        - Voice: pipe text_chunks through TTS

        After streaming completes, updates session history and fires
        background memory extraction on E-Cores.
        """
        if history is None:
            history = await self._get_history(session_id)

        request = KernelRequest(
            kind=JobKind.REPLY_TEXT.value,
            modality=JobModality.TEXT.value,
            user_id=os.getenv("AETHER_USER_ID", ""),
            session_id=session_id,
            payload={
                "text": text,
                "history": history,
                "enabled_plugins": self._plugin_context.loaded_plugins(),
                "pending_vision": vision,
            },
            priority=JobPriority.INTERACTIVE.value,
        )

        job_id = await self._scheduler.submit(request)

        collected_text: list[str] = []
        async for event in self._scheduler.stream(job_id):
            if event.stream_type == "text_chunk":
                collected_text.append(event.payload.get("text", ""))
            yield event

        # Update session history
        full_response = "".join(collected_text).strip()
        await self._save_turn(session_id, text, full_response)

        # Fire background memory extraction on E-Cores
        await self._submit_memory_extraction(text, full_response, session_id)

    async def generate_text_reply_session(
        self,
        text: str,
        session_id: str,
        history: list[dict] | None = None,
        vision: dict | None = None,
    ) -> AsyncGenerator[KernelEvent, None]:
        """P-worker text session entry point.

        User ingress must always go through the P-worker text handler
        (Gemini realtime session). There is no OpenRouter fallback here.
        """
        if self._text_reply_handler is None:
            raise RuntimeError("P-worker text handler unavailable")

        async for event in self._text_reply_handler(text, session_id, history, vision):
            yield event

    def set_text_reply_handler(
        self,
        handler: Callable[
            [str, str, list[dict] | None, dict | None],
            AsyncGenerator["KernelEvent", None],
        ]
        | None,
    ) -> None:
        """Set or clear custom text reply streaming handler."""
        self._text_reply_handler = handler

    async def generate_reply_voice(
        self,
        text: str,
        session_id: str,
    ) -> AsyncGenerator[KernelEvent, None]:
        """Compatibility shim for legacy VoiceSession implementations."""
        async for event in self.generate_reply(text=text, session_id=session_id):
            yield event

    # ─── Session History Helpers ─────────────────────────────────

    async def _get_history(self, session_id: str) -> list[dict]:
        """Get conversation history from SessionStore."""
        await self._session_store.ensure_session(session_id)
        return await self._session_store.get_messages_as_openai(session_id)

    async def _save_turn(
        self, session_id: str, user_text: str, assistant_text: str
    ) -> None:
        """Persist a user+assistant turn to SessionStore."""
        await self._session_store.ensure_session(session_id)
        await self._session_store.append_user_message(session_id, user_text)
        if assistant_text:
            await self._session_store.append_assistant_message(
                session_id, assistant_text
            )

    # ─── Session Loop (Autonomous Agent) ───────────────────────────

    async def run_session(
        self,
        session_id: str,
        user_message: str,
        enabled_plugins: list[str] | None = None,
        background: bool = False,
    ) -> str | None:
        """
        Run the autonomous agent loop for a session.

        This is the entry point for Scenario 2 (background tasks) and
        Scenario 3 (long-running autonomous work). It creates a SessionLoop
        and runs it until the agent decides it's done.

        Args:
            session_id: Session to run (created if needed)
            user_message: The user's request to process
            enabled_plugins: Plugin names for context building
            background: If True, run as a background task and return immediately.
                        The caller can monitor progress via EventBus.

        Returns:
            The final assistant text if run synchronously (background=False).
            The session_id if run as a background task (background=True).
        """
        if self._llm_core is None or self._context_builder is None:
            raise RuntimeError("run_session requires llm_core and context_builder")

        from aether.session.loop import SessionLoop

        # Ensure session exists and add the user message
        await self._session_store.ensure_session(session_id)
        await self._session_store.append_user_message(session_id, user_message)

        loop = SessionLoop(
            session_store=self._session_store,
            llm_core=self._llm_core,
            context_builder=self._context_builder,
            event_bus=self._event_bus,
        )

        if enabled_plugins is None:
            enabled_plugins = self._plugin_context.loaded_plugins()

        if background:
            # Run as a background task — return immediately
            abort = asyncio.Event()
            task = asyncio.create_task(
                loop.run(session_id, abort=abort, enabled_plugins=enabled_plugins)
            )
            self._session_tasks[session_id] = task

            # Clean up when done
            def _on_done(t: asyncio.Task) -> None:
                self._session_tasks.pop(session_id, None)

            task.add_done_callback(_on_done)

            logger.info("Session %s started in background", session_id)
            return session_id
        else:
            # Run synchronously — block until done
            return await loop.run(session_id, enabled_plugins=enabled_plugins)

    async def cancel_session_loop(self, session_id: str) -> bool:
        """Cancel a running background session loop.

        Returns True if a task was found and canceled, False otherwise.
        """
        task = self._session_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
            logger.info("Session loop %s canceled", session_id)
            return True
        return False

    def get_active_session_loops(self) -> list[str]:
        """Return session IDs with active background loops."""
        return [sid for sid, task in self._session_tasks.items() if not task.done()]

    # ─── Task Completion Listener ────────────────────────────────

    async def _listen_task_completed(self) -> None:
        """
        Listen for task.completed events from SubAgentManager and push
        notifications to WS sidecar clients.

        Runs as a background task for the lifetime of AgentCore.
        """
        if self._event_bus is None or self._task_completed_queue is None:
            return

        try:
            async for event in self._event_bus.listen(self._task_completed_queue):
                session_id = event.get("session_id", "unknown")
                logger.info("Task completed notification: %s", session_id)

                # Get the result text for the notification
                result_text = await self._session_store.get_last_assistant_text(
                    session_id
                )

                # Build notification
                preview = (result_text or "")[:200].strip()
                notification = {
                    "type": "task_completed",
                    "session_id": session_id,
                    "preview": preview or "(no output)",
                }

                await self.broadcast_notification(notification)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Task completed listener failed: %s", e, exc_info=True)

    # ─── Background Jobs ─────────────────────────────────────────

    async def _ensure_system_session(self) -> None:
        """Create the synthetic system session used by scheduled tasks."""
        if self._system_session_ready:
            return
        try:
            await self._session_store.create_session(
                session_id="system",
                agent_type="system",
                metadata={"kind": "internal"},
            )
            self._system_session_ready = True
        except Exception as e:
            logger.warning("Failed to ensure system session: %s", e)

    async def _submit_memory_extraction(
        self, user_text: str, assistant_text: str, session_id: str
    ) -> None:
        """Submit fact extraction via Task Ledger (background, non-blocking).

        Writes a MEMORY_EXTRACT task to the Task Ledger. The background
        memory extraction handler (_process_memory_extractions) picks it
        up and dispatches to the KernelScheduler for actual extraction.

        Single extraction path — the inline extraction in MemoryStore.add()
        has been removed to avoid duplicate fact extraction.
        """
        if not user_text or not assistant_text:
            return

        try:
            await self._task_ledger.submit(
                session_id=session_id,
                task_type=TaskType.MEMORY_EXTRACT.value,
                payload={
                    "user_message": user_text,
                    "assistant_message": assistant_text,
                },
                priority="low",
            )
        except Exception as e:
            # Background job — don't fail the reply
            logger.warning("Failed to submit memory extraction to ledger: %s", e)

    async def _process_memory_extractions(self) -> None:
        """Background handler: poll Task Ledger for MEMORY_EXTRACT tasks.

        Picks up pending MEMORY_EXTRACT tasks from the ledger and dispatches
        them to the KernelScheduler for actual extraction via MemoryService.
        Updates the ledger status on completion or failure.

        This bridges the Task Ledger (persistent record) with the
        KernelScheduler (execution engine). The ledger tracks status,
        the scheduler does the work.
        """
        poll_interval = 2.0  # seconds

        while True:
            try:
                task = await self._task_ledger.pick_next(
                    task_type=TaskType.MEMORY_EXTRACT.value,
                )
                if task is None:
                    await asyncio.sleep(poll_interval)
                    continue

                # Mark as running in the ledger
                await self._task_ledger.set_running(task.task_id)

                # Dispatch to KernelScheduler for actual extraction
                try:
                    job_id = await self._scheduler.submit(
                        KernelRequest(
                            kind=JobKind.MEMORY_FACT_EXTRACT.value,
                            modality=JobModality.SYSTEM.value,
                            user_id=os.getenv("AETHER_USER_ID", ""),
                            session_id=task.session_id,
                            payload=task.payload,
                            priority=JobPriority.BACKGROUND.value,
                        )
                    )

                    # Wait for the scheduler job to complete
                    async for _event in self._scheduler.stream(job_id):
                        pass  # Consume events — we just need completion

                    await self._task_ledger.set_complete(
                        task.task_id,
                        {
                            "extracted": True,
                        },
                    )

                except Exception as e:
                    await self._task_ledger.set_error(task.task_id, str(e))
                    logger.warning(
                        "Memory extraction task %s failed: %s",
                        task.task_id,
                        e,
                    )

            except asyncio.CancelledError:
                raise  # Let cancellation propagate for clean shutdown
            except Exception as e:
                logger.error("Memory extraction handler error: %s", e, exc_info=True)
                await asyncio.sleep(poll_interval)

    # ─── Nightly Analysis Loop ───────────────────────────────────

    async def _run_nightly_analysis_loop(self) -> None:
        """Run nightly analysis on a 24-hour timer.

        Submits a SCHEDULED task to the Task Ledger, then runs the
        NightlyAnalysisService. Candidate notifications from the analysis
        are queued to the notification table for the sweep to deliver.

        Runs every 24 hours from startup (Requirements.md §6.2).
        """
        assert self._nightly_service is not None

        interval = 24 * 60 * 60  # 24 hours
        # Wait 1 minute after startup before first check
        await asyncio.sleep(60)

        while True:
            task_id: str | None = None
            try:
                await self._ensure_system_session()

                # Write a SCHEDULED task to the Task Ledger for audit trail
                task_id = await self._task_ledger.submit(
                    session_id="system",
                    task_type=TaskType.SCHEDULED.value,
                    payload={"action": "nightly_analysis"},
                    priority="low",
                )
                await self._task_ledger.set_running(task_id)

                # Run the analysis
                result = await self._nightly_service.run_analysis()

                # Queue candidate notifications for delivery
                for notif in result.candidate_notifications:
                    try:
                        delivery_type = str(
                            notif.get("delivery_type", "surface")
                        ).strip()
                        delivery_hint = notif.get("deliver_at")
                        deliver_at = self._resolve_notification_deliver_at(
                            delivery_hint,
                            delivery_type=delivery_type,
                        )
                        await self._memory_store.queue_notification(
                            text=notif.get("text", ""),
                            delivery_type=delivery_type,
                            deliver_at=deliver_at,
                            source="proactive",
                            metadata={
                                "origin": "nightly_analysis",
                                "delivery_hint": delivery_hint,
                            },
                        )
                    except Exception as e:
                        logger.warning("Failed to queue proactive notification: %s", e)

                # Mark the ledger task as complete
                await self._task_ledger.set_complete(
                    task_id,
                    {
                        "new_decisions": result.new_decisions,
                        "notifications_queued": len(result.candidate_notifications),
                        "facts_consolidated": result.facts_consolidated,
                        "facts_flagged_stale": result.facts_flagged_stale,
                        "duration_ms": result.duration_ms,
                    },
                )

                logger.info(
                    "Nightly analysis completed (task=%s, duration=%dms)",
                    task_id,
                    result.duration_ms,
                )

            except asyncio.CancelledError:
                raise  # Let cancellation propagate for clean shutdown
            except Exception as e:
                logger.error("Nightly analysis failed: %s", e, exc_info=True)
                # Try to mark the ledger task as errored if we have a task_id
                if task_id is not None:
                    try:
                        await self._task_ledger.set_error(task_id, str(e))
                    except Exception:
                        pass

            await asyncio.sleep(interval)

    @classmethod
    def _resolve_notification_deliver_at(
        cls,
        delivery_hint: Any,
        *,
        delivery_type: str,
        now_ts: float | None = None,
    ) -> float | None:
        now_ts = now_ts or time.time()
        hint = str(delivery_hint or "").strip().lower()

        if hint in {"", "immediately", "now", "asap"}:
            if delivery_type in {
                "nudge",
                "surface",
                "deferred",
            } and cls._is_quiet_hours(now_ts):
                return cls._next_local_time(now_ts, hour=8, minute=0)
            return None

        if hint in {"morning", "tomorrow morning"}:
            return cls._next_local_time(now_ts, hour=8, minute=0)

        if hint in {"evening", "tonight"}:
            return cls._next_local_time(now_ts, hour=18, minute=0)

        relative = re.search(
            r"in\s+(\d+)\s*(minute|minutes|min|hour|hours|hr|hrs)", hint
        )
        if relative:
            amount = int(relative.group(1))
            unit = relative.group(2)
            delta = amount * 60 if unit.startswith("m") else amount * 3600
            return now_ts + delta

        before = re.search(r"before\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", hint)
        if before:
            hour = int(before.group(1))
            minute = int(before.group(2) or 0)
            meridiem = (before.group(3) or "").lower()
            if meridiem == "pm" and hour < 12:
                hour += 12
            if meridiem == "am" and hour == 12:
                hour = 0
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return cls._next_local_time(now_ts, hour=hour, minute=minute)

        return None

    @staticmethod
    def _next_local_time(now_ts: float, *, hour: int, minute: int) -> float:
        now = datetime.fromtimestamp(now_ts)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target.timestamp()

    @staticmethod
    def _is_quiet_hours(now_ts: float) -> bool:
        local_hour = datetime.fromtimestamp(now_ts).hour
        return local_hour >= 22 or local_hour < 7

    # ─── Notification Sweep ──────────────────────────────────────

    async def _sweep_notification_queue(self) -> None:
        """Sweep the notification queue every 60 seconds.

        Expires old notifications, then delivers any pending notifications
        that are ready (deliver_at <= now). Delivery goes through the
        existing notification subscriber pattern (WS sidecar).

        Runs every 60 seconds (Requirements.md §6.5).
        """
        while True:
            try:
                # Expire old notifications
                await self._memory_store.expire_old_notifications()

                # Get pending notifications ready for delivery
                pending = await self._memory_store.get_pending_notifications()
                metrics.gauge_set("service.notification.pending", len(pending))
                for notif in pending:
                    notif_id = int(notif.get("id", 0) or 0)
                    try:
                        if notif_id:
                            await self._memory_store.mark_delivery_attempt(notif_id)
                        await self._deliver_notification(notif)
                        await self._memory_store.mark_delivered(notif_id)
                    except Exception as e:
                        if notif_id:
                            await self._memory_store.mark_delivery_error(
                                notif_id, str(e)
                            )
                        metrics.inc(
                            "service.notification.delivery_error",
                            labels={
                                "delivery_type": str(
                                    notif.get("delivery_type", "unknown")
                                )
                            },
                        )
                        logger.warning(
                            "Failed to deliver notification %d: %s",
                            notif.get("id", 0),
                            e,
                        )

            except asyncio.CancelledError:
                raise  # Let cancellation propagate for clean shutdown
            except Exception as e:
                logger.error("Notification sweep failed: %s", e, exc_info=True)

            await asyncio.sleep(60)

    async def _deliver_notification(self, notif: dict) -> None:
        """Deliver a single notification through the subscriber pattern.

        Maps delivery_type to the appropriate delivery mechanism:
        - suppress: skip delivery entirely
        - queue: store only (already in DB), don't push
        - nudge: broadcast to WS sidecar (low priority)
        - surface: broadcast to WS sidecar (normal priority)
        - interrupt: speak through voice + broadcast to WS sidecar
        """
        delivery_type = notif.get("delivery_type", "surface")
        text = notif.get("text", "")

        if not text or delivery_type == "suppress":
            return

        if delivery_type == "queue":
            # Already persisted in the notification table — no push needed
            return

        if delivery_type == "interrupt":
            # Deliver through voice if available, always broadcast to WS
            await self.speak_notification(text)
            self._record_notification_delivery_metric(
                notif,
                delivery_type=delivery_type,
                channel="voice_and_ws",
            )
        else:
            # nudge or surface — broadcast to WS sidecar
            level = "nudge" if delivery_type == "nudge" else "speak"
            await self.broadcast_notification(
                {
                    "type": "proactive",
                    "level": level,
                    "text": text,
                    "source": notif.get("source", "proactive"),
                    "notification_id": notif.get("id"),
                }
            )
            self._record_notification_delivery_metric(
                notif,
                delivery_type=delivery_type,
                channel="ws",
            )

    @staticmethod
    def _delivery_latency_ms(
        created_at: Any,
        *,
        now_ts: float | None = None,
    ) -> float | None:
        if not isinstance(created_at, (int, float)):
            return None
        now = now_ts if now_ts is not None else time.time()
        return max(0.0, (now - float(created_at)) * 1000.0)

    def _record_notification_delivery_metric(
        self,
        notif: dict,
        *,
        delivery_type: str,
        channel: str,
    ) -> None:
        latency_ms = self._delivery_latency_ms(notif.get("created_at"))
        labels = {"delivery_type": delivery_type, "channel": channel}
        if latency_ms is not None:
            metrics.observe(
                "service.notification.delivery_ms", latency_ms, labels=labels
            )
        metrics.inc("service.notification.delivered", labels=labels)

    # ─── Greeting ────────────────────────────────────────────────

    async def generate_greeting(
        self,
        session_id: str,
        is_resume: bool = False,
    ) -> str | None:
        """
        Generate a contextual, temporal greeting for voice sessions.

        Rules:
        - Fast reconnect/resume: silent continuation.
        - No conversation context and no meaningful briefing: silent.
        - Morning long-gap reconnect: short briefing with notifications/weather.
        - Otherwise: contextual welcome based on recent memory/session history.
        """
        now = time.time()

        # Determine gap since last session
        last_session_end = await self._memory_store.get_last_session_end()
        if last_session_end:
            gap_seconds = now - last_session_end
        elif self._last_greeting_at > 0:
            gap_seconds = now - self._last_greeting_at
        else:
            gap_seconds = float("inf")  # First time ever

        self._last_greeting_at = now
        gap_minutes = gap_seconds / 60.0

        logger.info(
            "Greeting gap: %.1f min (last_session_end=%s)",
            gap_minutes,
            last_session_end,
        )

        # Fast reconnect/resume: continue silently where user left off.
        if is_resume and gap_minutes < 30:
            logger.info("Greeting: silent resume (gap < 30 min)")
            return None

        history = await self._get_history(session_id)
        has_conversation = bool(history)

        # Extract known user name for greeting personalization.
        name = None
        try:
            facts = await self._memory_store.get_facts()
            name = self._extract_name_from_facts(facts)
        except Exception:
            facts = []

        # Build a compact morning briefing when meaningful.
        briefing_text, has_weather, has_notifications = self._build_briefing_text(now)

        from datetime import datetime

        hour = datetime.now().hour
        is_morning = 5 <= hour < 12
        if is_morning and gap_seconds > 6 * 3600 and (has_weather or has_notifications):
            who = f", {name}" if name else ""
            greeting = f"Good morning{who}. {briefing_text}".strip()
            logger.info("Greeting: morning briefing")
            return greeting

        # If we have no context and no briefing value, remain silent.
        if not has_conversation and not has_notifications and not has_weather:
            logger.info("Greeting: silent (no prior conversation context)")
            return None

        # Medium gap: contextual continuation from recent session summary.
        if 30 <= gap_minutes < 8 * 60:
            try:
                sessions = await self._memory_store.get_session_summaries(limit=1)
                if sessions:
                    summary = str(sessions[0].get("summary", "")).strip()
                    if summary:
                        summary = re.sub(r"\s+", " ", summary)
                        summary = summary[:120].rstrip(" ,.;")
                        prefix = f"Welcome back{', ' + name if name else ''}."
                        return f"{prefix} Last time: {summary}."
            except Exception:
                pass

        # If we have briefing items but it is not a long-gap morning, keep it short.
        if has_notifications or has_weather:
            who = f", {name}" if name else ""
            return f"Welcome back{who}. {briefing_text}".strip()

        # Final fallback: memory-aware LLM greeting.
        try:
            from aether.greeting import generate_greeting

            greeting = await generate_greeting(
                memory=self._memory_store,
                llm_provider=self._llm_provider,
            )
            if not greeting or greeting.lower() in {
                "welcome back",
                "welcome back.",
                "hello",
                "hello.",
                "hi",
                "hi.",
            }:
                return None
            logger.info("Voice greeting: '%s'", greeting[:80])
            return greeting
        except Exception as e:
            logger.error("Greeting generation failed: %s", e)
            return None

    # ─── Session Management ──────────────────────────────────────

    async def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        return await self._get_history(session_id)

    async def cancel_session(self, session_id: str) -> int:
        """Cancel all pending/running jobs for a session (on disconnect)."""
        return await self._scheduler.cancel_by_session(session_id)

    # ─── Notification Subscribers (WS Sidecar) ───────────────────

    def subscribe_notifications(self, callback: Callable) -> None:
        """Register a WS sidecar connection for push notifications."""
        self._notification_subscribers.append(callback)

    def unsubscribe_notifications(self, callback: Callable) -> None:
        """Unregister a WS sidecar connection."""
        try:
            self._notification_subscribers.remove(callback)
        except ValueError:
            pass

    async def broadcast_notification(self, notification: dict) -> None:
        """Push a notification to all subscribed WS sidecar connections."""
        self._record_briefing_item(notification)
        for cb in self._notification_subscribers:
            try:
                await cb(notification)
            except Exception:
                logger.debug(
                    "Notification broadcast failed for subscriber", exc_info=True
                )

    def _record_briefing_item(self, notification: dict[str, Any]) -> None:
        text = str(notification.get("text", "")).strip()
        if not text:
            return
        level = str(notification.get("level", "")).lower()
        if level not in {"speak", "nudge", "batch"}:
            return

        kind = "weather" if self._looks_like_weather(text) else "notification"
        self._briefing_items.append({"ts": time.time(), "kind": kind, "text": text})

        # Keep only recent compact window.
        cutoff = time.time() - (24 * 3600)
        self._briefing_items = [i for i in self._briefing_items if i["ts"] >= cutoff][
            -30:
        ]

    def _build_briefing_text(self, now: float) -> tuple[str, bool, bool]:
        cutoff = now - (12 * 3600)
        items = [i for i in self._briefing_items if i["ts"] >= cutoff]
        weather_items = [i for i in items if i["kind"] == "weather"]
        notif_items = [i for i in items if i["kind"] != "weather"]

        parts: list[str] = []
        has_weather = bool(weather_items)
        has_notifications = bool(notif_items)

        if has_notifications:
            n = len(notif_items)
            parts.append(f"You have {n} new notification{'s' if n != 1 else ''}.")
        if has_weather:
            latest_weather = str(weather_items[-1]["text"]).strip()
            latest_weather = re.sub(r"\s+", " ", latest_weather)
            parts.append(f"Weather update: {latest_weather[:120].rstrip(' ,.;')}.")

        return " ".join(parts).strip(), has_weather, has_notifications

    def _extract_name_from_facts(self, facts: list[str]) -> str | None:
        for fact in facts[:20]:
            text = fact.strip().rstrip(".")
            m = re.search(
                r"user(?:'s)?\s+name\s+is\s+([A-Za-z][A-Za-z\-']{1,30})",
                text,
                re.IGNORECASE,
            )
            if m:
                return m.group(1)
        return None

    def _looks_like_weather(self, text: str) -> bool:
        low = text.lower()
        weather_terms = (
            "weather",
            "forecast",
            "temperature",
            "rain",
            "sunny",
            "cloud",
            "wind",
            "humidity",
            "degrees",
        )
        return any(t in low for t in weather_terms)

    # ─── Spoken Notification Delivery ────────────────────────────

    async def speak_notification(self, text: str) -> bool:
        """Deliver a notification via TTS through active voice sessions.

        Returns True if delivered to at least one session, False otherwise.
        Always also broadcasts to WS sidecar as fallback.
        """
        delivered = False

        if self._voice_transport is not None:
            try:
                sessions = self._voice_transport.get_active_sessions()
                for session in sessions:
                    if hasattr(session, "inject_text"):
                        await session.inject_text(text)
                    else:
                        await session.deliver_notification(text)
                    delivered = True
            except Exception as e:
                logger.warning("Spoken notification delivery failed: %s", e)

        # Always broadcast to WS sidecar as fallback
        await self.broadcast_notification({"level": "speak", "text": text})

        return delivered

    # ─── Health ──────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Return AgentCore + scheduler health."""
        scheduler_health = await self._scheduler.health_check()

        sessions = await self._session_store.list_sessions(limit=1000)
        session_count = len(sessions)

        return {
            "agent_core": True,
            "sessions": session_count,
            "session_backend": "sqlite",
            "active_session_loops": self.get_active_session_loops(),
            "last_greeting_at": self._last_greeting_at,
            "notification_subscribers": len(self._notification_subscribers),
            "scheduler": scheduler_health,
        }
