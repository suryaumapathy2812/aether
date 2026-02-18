"""
Standard event format for all plugins.

Every plugin normalizes its raw webhook payloads into PluginEvent.
This enables the decision engine to reason uniformly across Gmail, Slack, WhatsApp, etc.
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field


@dataclass
class PluginEvent:
    """Unified inbound event from any plugin sensor."""

    # Identity
    plugin: str                         # "gmail", "slack", "whatsapp"
    event_type: str                     # "message.new", "thread.updated"
    source_id: str                      # Original ID from the service

    # Content
    summary: str                        # One-line: "Email from brother: Friday plans?"
    content: str = ""                   # Full body (email text, message body, etc.)
    sender: dict = field(default_factory=dict)  # {"name": "John", "email": "john@..."}

    # Decision hints (plugin provides best-guess, engine refines)
    urgency: str = "low"                # "low" | "medium" | "high"
    category: str = "general"           # "personal" | "work" | "notification" | "marketing"
    requires_action: bool = False       # Does this need a response?

    # What the LLM can do with this event
    available_actions: list[str] = field(default_factory=list)  # ["read_gmail", "send_reply"]

    # Plugin-specific extras
    metadata: dict = field(default_factory=dict)

    # Auto-generated
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plugin": self.plugin,
            "event_type": self.event_type,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "content": self.content,
            "sender": self.sender,
            "urgency": self.urgency,
            "category": self.category,
            "requires_action": self.requires_action,
            "available_actions": self.available_actions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PluginEvent:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:16]),
            plugin=data["plugin"],
            event_type=data["event_type"],
            source_id=data.get("source_id", ""),
            timestamp=data.get("timestamp", time.time()),
            summary=data.get("summary", ""),
            content=data.get("content", ""),
            sender=data.get("sender", {}),
            urgency=data.get("urgency", "low"),
            category=data.get("category", "general"),
            requires_action=data.get("requires_action", False),
            available_actions=data.get("available_actions", []),
            metadata=data.get("metadata", {}),
        )
