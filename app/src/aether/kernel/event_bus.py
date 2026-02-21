"""
Event Bus — lightweight async pub/sub for internal events.

Enables decoupled communication between components:
- SessionLoop publishes events as it processes (text chunks, tool results, status)
- Sub-agents publish completion events
- AgentCore subscribes to task.completed to push notifications
- SSE endpoints subscribe to session events for real-time streaming

Design:
- Topic-based: publishers write to topics, subscribers listen on topics
- Each subscriber gets its own asyncio.Queue (no cross-talk)
- Non-blocking: publish() never blocks the publisher
- Cleanup: unsubscribe() removes the queue

Topics follow a convention:
- session.{session_id}.event  — all events for a session
- session.{session_id}.status — status changes (idle/busy/done/error)
- task.completed              — any sub-agent task completed

Usage:
    bus = EventBus()

    # Subscribe
    queue = bus.subscribe("session.abc.event")
    async for event in bus.listen(queue):
        print(event)

    # Publish (from another coroutine)
    await bus.publish("session.abc.event", {"type": "text_chunk", "text": "hello"})

    # Cleanup
    bus.unsubscribe("session.abc.event", queue)
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)

# Sentinel to signal end of stream
_STREAM_END = object()


class EventBus:
    """
    Lightweight async pub/sub event bus.

    Thread-safe via asyncio (single event loop). Each subscriber gets
    its own Queue so slow consumers don't block fast ones.
    """

    def __init__(self) -> None:
        # topic → list of subscriber queues
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

    async def publish(self, topic: str, event: Any) -> int:
        """
        Publish an event to all subscribers of a topic.

        Non-blocking: uses put_nowait so the publisher never waits.
        Returns the number of subscribers that received the event.
        """
        queues = self._subscribers.get(topic, [])
        delivered = 0
        for queue in queues:
            try:
                queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning(
                    "Event bus: subscriber queue full for topic %s, dropping event",
                    topic,
                )
        return delivered

    def subscribe(
        self,
        topic: str,
        maxsize: int = 1000,
    ) -> asyncio.Queue:
        """
        Subscribe to a topic. Returns a Queue that receives events.

        The caller should use listen() to iterate over events, or
        read from the queue directly.

        Args:
            topic: The topic to subscribe to
            maxsize: Maximum queue size (events dropped if full)

        Returns:
            An asyncio.Queue that will receive published events
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers[topic].append(queue)
        logger.debug(
            "Subscribed to topic: %s (total: %d)", topic, len(self._subscribers[topic])
        )
        return queue

    def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        """
        Remove a subscriber's queue from a topic.

        Safe to call even if the queue was already removed.
        """
        queues = self._subscribers.get(topic, [])
        try:
            queues.remove(queue)
            if not queues:
                del self._subscribers[topic]
            logger.debug("Unsubscribed from topic: %s", topic)
        except ValueError:
            pass

    async def publish_end(self, topic: str) -> None:
        """
        Signal end-of-stream to all subscribers of a topic.

        Subscribers using listen() will stop iterating.
        """
        queues = self._subscribers.get(topic, [])
        for queue in queues:
            try:
                queue.put_nowait(_STREAM_END)
            except asyncio.QueueFull:
                pass

    async def listen(self, queue: asyncio.Queue) -> AsyncGenerator[Any, None]:
        """
        Async generator that yields events from a subscriber queue.

        Stops when _STREAM_END sentinel is received or the queue is empty
        and no more events are expected.

        Usage:
            queue = bus.subscribe("session.abc.event")
            async for event in bus.listen(queue):
                handle(event)
        """
        while True:
            item = await queue.get()
            if item is _STREAM_END:
                break
            yield item

    def subscriber_count(self, topic: str) -> int:
        """Number of active subscribers for a topic."""
        return len(self._subscribers.get(topic, []))

    def active_topics(self) -> list[str]:
        """List all topics with at least one subscriber."""
        return [t for t, subs in self._subscribers.items() if subs]

    def clear_topic(self, topic: str) -> None:
        """Remove all subscribers for a topic. Used for cleanup."""
        queues = self._subscribers.pop(topic, [])
        for queue in queues:
            try:
                queue.put_nowait(_STREAM_END)
            except asyncio.QueueFull:
                pass
