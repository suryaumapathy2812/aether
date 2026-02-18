"""
Event Processor — Aether's decision engine.

Receives PluginEvents from sensors (email, slack, etc.), decides what to
surface to the user based on urgency, preferences, and context.

Decision outcomes:
  - "surface"         → speak notification to user via TTS
  - "archive"         → store silently, don't interrupt
  - "action_required" → speak notification + suggest available actions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from aether.plugins.event import PluginEvent

log = logging.getLogger("aether.event")

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
If the event is informational but not time-sensitive, prefer "deferred" over "surface".

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


@dataclass
class EventDecision:
    """Result of the decision engine processing an event."""

    action: str  # "surface" | "archive" | "action_required" | "deferred"
    notification: str  # Natural language notification (empty if archived)
    event: PluginEvent


class EventProcessor:
    """
    Decision engine for plugin events.

    Uses LLM to classify events and generate notifications.
    Learns from user preferences stored in memory.
    """

    def __init__(self, llm_provider, memory_store):
        self.llm = llm_provider
        self.memory = memory_store

    async def process(self, event: PluginEvent) -> EventDecision:
        """
        Process an inbound plugin event.

        1. Query memory for relevant user preferences
        2. Ask LLM to classify: surface / archive / action_required
        3. If surfacing, generate natural language notification
        4. Return decision
        """
        # 1. Get relevant preferences from memory
        preferences = await self._get_preferences(event)

        # 2. Classify
        action = await self._classify(event, preferences)
        log.info(f"Event decision: {event.plugin}/{event.event_type} → {action}")

        # 3. Generate notification if surfacing or deferred
        # (deferred notifications are used at flush time)
        notification = ""
        if action in ("surface", "action_required", "deferred"):
            notification = await self._generate_notification(event)

        return EventDecision(action=action, notification=notification, event=event)

    async def _get_preferences(self, event: PluginEvent) -> str:
        """Query memory for user preferences related to this event."""
        try:
            # Search for preferences about this plugin/sender
            sender_name = event.sender.get("name", "")
            sender_email = event.sender.get("email", "")
            query = f"{event.plugin} {event.event_type} {sender_name} {sender_email} notification preference"

            results = await self.memory.search(query, limit=5)
            if results:
                # Format relevant memories
                lines = []
                for r in results:
                    if r.get("type") == "fact":
                        lines.append(f"- {r['content']}")
                    elif r.get("type") == "action":
                        lines.append(f"- Previously: {r['content']}")
                return "\n".join(lines) if lines else "No specific preferences found."
            return "No specific preferences found."
        except Exception as e:
            log.debug(f"Preference lookup failed: {e}")
            return "No specific preferences found."

    async def _classify(self, event: PluginEvent, preferences: str) -> str:
        """Use LLM to decide: surface, archive, or action_required."""
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

        try:
            messages = [{"role": "user", "content": prompt}]
            response = ""
            async for event_chunk in self.llm.generate_stream(
                messages, max_tokens=10, temperature=0.1
            ):
                if hasattr(event_chunk, "content"):
                    response += event_chunk.content
                elif isinstance(event_chunk, str):
                    response += event_chunk

            decision = (
                response.strip().lower().split()[0] if response.strip() else "archive"
            )

            if decision in ("surface", "archive", "action_required", "deferred"):
                return decision
            return "archive"  # Default to not interrupting

        except Exception as e:
            log.error(f"Classification failed: {e}")
            # Fallback: surface high urgency, archive everything else
            if event.urgency == "high" or event.requires_action:
                return "surface"
            return "archive"

    async def _generate_notification(self, event: PluginEvent) -> str:
        """Generate a natural language notification for the user."""
        sender_name = event.sender.get("name", event.sender.get("email", "someone"))

        prompt = NOTIFICATION_PROMPT.format(
            summary=event.summary,
            sender_name=sender_name,
            plugin=event.plugin,
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = ""
            async for event_chunk in self.llm.generate_stream(
                messages, max_tokens=60, temperature=0.7
            ):
                if hasattr(event_chunk, "content"):
                    response += event_chunk.content
                elif isinstance(event_chunk, str):
                    response += event_chunk

            return response.strip().strip('"')

        except Exception as e:
            log.error(f"Notification generation failed: {e}")
            return event.summary  # Fallback to raw summary
