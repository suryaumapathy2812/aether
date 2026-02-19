"""
Notification Service — event classification and notification compose.

Handles notification_* job kinds:
- notification_decide: Classify an event (surface/archive/action_required/deferred)
- notification_compose: Generate natural language notification text

Wraps the existing EventProcessor logic but routes all LLM calls
through LLMCore for consistent contracts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import time

from aether.core.metrics import metrics
from aether.llm.contracts import LLMEventType, LLMRequestEnvelope

if TYPE_CHECKING:
    from aether.llm.core import LLMCore
    from aether.memory.store import MemoryStore
    from aether.plugins.event import PluginEvent

logger = logging.getLogger(__name__)

# Decision prompt — same as EventProcessor but routed through LLMCore
DECISION_PROMPT = """You are Aether's notification filter. Given an inbound event and the user's known preferences, decide whether to notify the user.

EVENT:
  Plugin: {plugin}
  Type: {event_type}
  From: {sender}
  Summary: {summary}
  Urgency: {urgency}
  Requires action: {requires_action}

USER PREFERENCES (from memory):
{preferences}

RULES:
- "surface" = tell the user about this (speak it) — for things worth mentioning now
- "archive" = store silently, don't interrupt — for things you can check later
- "action_required" = tell the user AND suggest they take action — for things needing immediate response
- "deferred" = worth telling the user, but not right now — batch with others and notify later

If the user has said to ignore this type of event or sender, choose "archive".
If the sender is someone important to the user (family, close contacts), lean toward "surface".
If urgency is "high" or requires_action is true, lean toward "action_required".
Low-urgency, non-actionable content (newsletters, social updates, FYI emails) → "deferred".

Respond with ONLY one word: surface, archive, action_required, or deferred"""

NOTIFICATION_PROMPT = """You are Aether, a personal AI assistant. Compose a brief, natural spoken notification for the user based on this event. Keep it under 2 sentences. Be warm and concise — like a thoughtful assistant whispering in your ear.

Event: {summary}
From: {sender_name}
Plugin: {plugin}

Examples of good notifications:
- "Your brother wants to know if you're free Friday night."
- "New email from Sarah about the project deadline."
- "Slack message from the team — they're asking about the deployment."

Notification:"""

VALID_DECISIONS = {"surface", "archive", "action_required", "deferred"}


@dataclass
class NotificationDecision:
    """Result of the notification decision engine."""

    action: str  # surface, archive, action_required, deferred
    notification: str  # Natural language notification (empty if archived)
    event: Any  # The original PluginEvent


class NotificationService:
    """
    Event classification and notification generation through LLMCore.

    Replaces direct LLM calls in EventProcessor with LLMCore calls,
    ensuring all LLM usage goes through the shared interface.
    """

    def __init__(
        self,
        llm_core: "LLMCore",
        memory_store: "MemoryStore",
    ) -> None:
        self._llm_core = llm_core
        self._memory_store = memory_store

    async def process_event(self, event: "PluginEvent") -> NotificationDecision:
        """
        Process an inbound plugin event.

        1. Query memory for relevant user preferences
        2. Classify via LLMCore: surface / archive / action_required / deferred
        3. If surfacing, generate natural language notification via LLMCore
        4. Return decision

        Args:
            event: The plugin event to process

        Returns:
            NotificationDecision with action and notification text
        """
        started = time.time()

        # 1. Get relevant preferences from memory
        preferences = await self._get_preferences(event)

        # 2. Classify
        action = await self._classify(event, preferences)
        logger.info(f"Event decision: {event.plugin}/{event.event_type} → {action}")

        # 3. Generate notification if surfacing
        notification = ""
        if action in ("surface", "action_required", "deferred"):
            notification = await self._generate_notification(event)

        elapsed_ms = round((time.time() - started) * 1000)
        metrics.observe("service.notification.decision_ms", elapsed_ms)
        metrics.inc(
            "service.notification.processed",
            labels={"decision": action, "plugin": event.plugin},
        )

        return NotificationDecision(
            action=action, notification=notification, event=event
        )

    async def _get_preferences(self, event: "PluginEvent") -> str:
        """Query memory for user preferences related to this event."""
        try:
            sender_name = event.sender.get("name", "")
            sender_email = event.sender.get("email", "")
            query = (
                f"{event.plugin} {event.event_type} "
                f"{sender_name} {sender_email} notification preference"
            )

            results = await self._memory_store.search(query, limit=5)
            if results:
                lines = []
                for r in results:
                    if r.get("type") == "fact":
                        lines.append(f"- {r['content']}")
                    elif r.get("type") == "action":
                        lines.append(f"- Previously: {r['content']}")
                return "\n".join(lines) if lines else "No specific preferences found."
            return "No specific preferences found."
        except Exception as e:
            logger.debug(f"Preference lookup failed: {e}")
            return "No specific preferences found."

    async def _classify(self, event: "PluginEvent", preferences: str) -> str:
        """Use LLMCore to classify the event."""
        sender_str = event.sender.get("name", event.sender.get("email", "unknown"))

        prompt = DECISION_PROMPT.format(
            plugin=event.plugin,
            event_type=event.event_type,
            sender=sender_str,
            summary=event.summary,
            urgency=event.urgency,
            requires_action=event.requires_action,
            preferences=preferences,
        )

        envelope = LLMRequestEnvelope(
            kind="notification_decide",
            modality="system",
            messages=[{"role": "user", "content": prompt}],
            policy={"max_tokens": 10, "temperature": 0.1},
        )

        try:
            response = await self._collect_response(envelope)
            decision = (
                response.strip().lower().split()[0] if response.strip() else "archive"
            )

            if decision in VALID_DECISIONS:
                return decision
            return "archive"

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Fallback: surface high urgency, archive everything else
            if event.urgency == "high" or event.requires_action:
                return "surface"
            return "archive"

    async def _generate_notification(self, event: "PluginEvent") -> str:
        """Generate a natural language notification via LLMCore."""
        sender_name = event.sender.get("name", event.sender.get("email", "someone"))

        prompt = NOTIFICATION_PROMPT.format(
            summary=event.summary,
            sender_name=sender_name,
            plugin=event.plugin,
        )

        envelope = LLMRequestEnvelope(
            kind="notification_compose",
            modality="system",
            messages=[{"role": "user", "content": prompt}],
            policy={"max_tokens": 60, "temperature": 0.7},
        )

        try:
            response = await self._collect_response(envelope)
            return response.strip().strip('"')
        except Exception as e:
            logger.error(f"Notification generation failed: {e}")
            return event.summary  # Fallback to raw summary

    async def _collect_response(self, envelope: LLMRequestEnvelope) -> str:
        """Collect full text response from LLMCore."""
        chunks: list[str] = []

        async for event in self._llm_core.generate_with_tools(envelope):
            if event.event_type == LLMEventType.TEXT_CHUNK.value:
                chunks.append(event.payload.get("text", ""))

        return " ".join(chunks)
