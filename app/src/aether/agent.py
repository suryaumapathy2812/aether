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
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, AsyncGenerator, Callable

from aether.kernel.contracts import (
    JobKind,
    JobModality,
    JobPriority,
    KernelEvent,
    KernelRequest,
)

if TYPE_CHECKING:
    from aether.kernel.scheduler import KernelScheduler
    from aether.memory.store import MemoryStore
    from aether.plugins.context import PluginContextStore
    from aether.providers.base import LLMProvider
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
    ) -> None:
        self._scheduler = scheduler
        self._memory_store = memory_store
        self._llm_provider = llm_provider
        self._tool_registry = tool_registry
        self._skill_loader = skill_loader
        self._plugin_context = plugin_context

        # Conversation history per session (in-memory, single user)
        self._sessions: dict[str, list[dict]] = {}

        # Notification subscribers (WS sidecar connections)
        self._notification_subscribers: list[Callable] = []

        # Voice greeting flag — only greet on first WebRTC connection
        self._has_greeted = False

    # ─── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Start the scheduler (worker pools)."""
        await self._scheduler.start()
        logger.info("AgentCore started")

    async def stop(self) -> None:
        """Stop the scheduler and clean up."""
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
            history = self._sessions.get(session_id, [])

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
        session_history = self._sessions.setdefault(session_id, [])
        session_history.append({"role": "user", "content": text})
        if full_response:
            session_history.append({"role": "assistant", "content": full_response})

        # Fire background memory extraction on E-Cores
        await self._submit_memory_extraction(text, full_response, session_id)

    async def generate_reply_voice(
        self,
        text: str,
        session_id: str,
    ) -> AsyncGenerator[KernelEvent, None]:
        """
        Same as generate_reply but tagged as voice modality.

        The scheduler routes this to ReplyService with mode="voice",
        which adjusts context (shorter system prompt, voice-friendly output).
        """
        history = self._sessions.get(session_id, [])

        request = KernelRequest(
            kind=JobKind.REPLY_VOICE.value,
            modality=JobModality.VOICE.value,
            user_id=os.getenv("AETHER_USER_ID", ""),
            session_id=session_id,
            payload={
                "text": text,
                "history": history,
                "enabled_plugins": self._plugin_context.loaded_plugins(),
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
        session_history = self._sessions.setdefault(session_id, [])
        session_history.append({"role": "user", "content": text})
        if full_response:
            session_history.append({"role": "assistant", "content": full_response})

        # Fire background memory extraction on E-Cores
        await self._submit_memory_extraction(text, full_response, session_id)

    # ─── Background Jobs ─────────────────────────────────────────

    async def _submit_memory_extraction(
        self, user_text: str, assistant_text: str, session_id: str
    ) -> None:
        """Submit fact extraction to E-Cores (background, non-blocking)."""
        if not user_text or not assistant_text:
            return

        try:
            await self._scheduler.submit(
                KernelRequest(
                    kind=JobKind.MEMORY_FACT_EXTRACT.value,
                    modality=JobModality.SYSTEM.value,
                    user_id=os.getenv("AETHER_USER_ID", ""),
                    session_id=session_id,
                    payload={
                        "user_message": user_text,
                        "assistant_message": assistant_text,
                    },
                    priority=JobPriority.BACKGROUND.value,
                )
            )
        except Exception as e:
            # Background job — don't fail the reply
            logger.warning("Failed to submit memory extraction: %s", e)

    # ─── Greeting ────────────────────────────────────────────────

    async def generate_greeting(self) -> str | None:
        """
        Generate a personalized voice greeting.

        Returns greeting text on first call, None on subsequent calls.
        Only used by WebRTC voice — not HTTP text chat.
        """
        if self._has_greeted:
            return None
        self._has_greeted = True

        try:
            from aether.greeting import generate_greeting

            greeting = await generate_greeting(
                memory=self._memory_store,
                llm_provider=self._llm_provider,
            )
            logger.info("Voice greeting: '%s'", greeting[:80])
            return greeting
        except Exception as e:
            logger.error("Greeting generation failed: %s", e)
            return "Hey there, good to see you."

    # ─── Session Management ──────────────────────────────────────

    def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        return self._sessions.get(session_id, [])

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._sessions.pop(session_id, None)

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
        for cb in self._notification_subscribers:
            try:
                await cb(notification)
            except Exception:
                logger.debug(
                    "Notification broadcast failed for subscriber", exc_info=True
                )

    # ─── Health ──────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Return AgentCore + scheduler health."""
        scheduler_health = await self._scheduler.health_check()
        return {
            "agent_core": True,
            "sessions": len(self._sessions),
            "has_greeted": self._has_greeted,
            "notification_subscribers": len(self._notification_subscribers),
            "scheduler": scheduler_health,
        }
