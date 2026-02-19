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
import re
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable

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

        # Voice transport reference for spoken notifications
        self._voice_transport: Any = None

        # Track last greeting time for time-gap logic
        self._last_greeting_at: float = 0.0

        # Briefing queue for temporal greetings (notifications/weather)
        self._briefing_items: list[dict[str, Any]] = []

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

        history = self._sessions.get(session_id, [])
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
        return {
            "agent_core": True,
            "sessions": len(self._sessions),
            "last_greeting_at": self._last_greeting_at,
            "notification_subscribers": len(self._notification_subscribers),
            "scheduler": scheduler_health,
        }
