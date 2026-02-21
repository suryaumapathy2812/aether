"""Tests for EventBus — async pub/sub."""

import asyncio

import pytest

from aether.kernel.event_bus import EventBus


@pytest.mark.asyncio
async def test_publish_and_listen():
    bus = EventBus()
    queue = bus.subscribe("test.topic")

    await bus.publish("test.topic", {"type": "hello"})
    await bus.publish_end("test.topic")

    events = []
    async for event in bus.listen(queue):
        events.append(event)

    assert len(events) == 1
    assert events[0] == {"type": "hello"}


@pytest.mark.asyncio
async def test_multiple_subscribers():
    bus = EventBus()
    q1 = bus.subscribe("test.topic")
    q2 = bus.subscribe("test.topic")

    delivered = await bus.publish("test.topic", "event-1")
    assert delivered == 2

    await bus.publish_end("test.topic")

    events1 = [e async for e in bus.listen(q1)]
    events2 = [e async for e in bus.listen(q2)]

    assert events1 == ["event-1"]
    assert events2 == ["event-1"]


@pytest.mark.asyncio
async def test_topic_isolation():
    bus = EventBus()
    q_a = bus.subscribe("topic.a")
    q_b = bus.subscribe("topic.b")

    await bus.publish("topic.a", "event-a")
    await bus.publish("topic.b", "event-b")
    await bus.publish_end("topic.a")
    await bus.publish_end("topic.b")

    events_a = [e async for e in bus.listen(q_a)]
    events_b = [e async for e in bus.listen(q_b)]

    assert events_a == ["event-a"]
    assert events_b == ["event-b"]


@pytest.mark.asyncio
async def test_unsubscribe():
    bus = EventBus()
    queue = bus.subscribe("test.topic")
    assert bus.subscriber_count("test.topic") == 1

    bus.unsubscribe("test.topic", queue)
    assert bus.subscriber_count("test.topic") == 0


@pytest.mark.asyncio
async def test_unsubscribe_idempotent():
    bus = EventBus()
    queue = bus.subscribe("test.topic")
    bus.unsubscribe("test.topic", queue)
    bus.unsubscribe("test.topic", queue)  # Should not raise


@pytest.mark.asyncio
async def test_publish_to_empty_topic():
    bus = EventBus()
    delivered = await bus.publish("nonexistent", "event")
    assert delivered == 0


@pytest.mark.asyncio
async def test_subscriber_count():
    bus = EventBus()
    assert bus.subscriber_count("test") == 0

    q1 = bus.subscribe("test")
    assert bus.subscriber_count("test") == 1

    q2 = bus.subscribe("test")
    assert bus.subscriber_count("test") == 2

    bus.unsubscribe("test", q1)
    assert bus.subscriber_count("test") == 1


@pytest.mark.asyncio
async def test_active_topics():
    bus = EventBus()
    assert bus.active_topics() == []

    bus.subscribe("topic.a")
    bus.subscribe("topic.b")
    assert set(bus.active_topics()) == {"topic.a", "topic.b"}


@pytest.mark.asyncio
async def test_clear_topic():
    bus = EventBus()
    q1 = bus.subscribe("test.topic")
    q2 = bus.subscribe("test.topic")

    bus.clear_topic("test.topic")
    assert bus.subscriber_count("test.topic") == 0

    # Queues should receive end sentinel
    events1 = [e async for e in bus.listen(q1)]
    events2 = [e async for e in bus.listen(q2)]
    assert events1 == []
    assert events2 == []


@pytest.mark.asyncio
async def test_multiple_events_in_order():
    bus = EventBus()
    queue = bus.subscribe("test.topic")

    for i in range(5):
        await bus.publish("test.topic", f"event-{i}")
    await bus.publish_end("test.topic")

    events = [e async for e in bus.listen(queue)]
    assert events == [f"event-{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_concurrent_publish_and_listen():
    """Test that publish and listen work concurrently."""
    bus = EventBus()
    queue = bus.subscribe("test.topic")
    received = []

    async def publisher():
        for i in range(10):
            await bus.publish("test.topic", f"event-{i}")
            await asyncio.sleep(0.001)
        await bus.publish_end("test.topic")

    async def listener():
        async for event in bus.listen(queue):
            received.append(event)

    await asyncio.gather(publisher(), listener())
    assert len(received) == 10
    assert received == [f"event-{i}" for i in range(10)]


@pytest.mark.asyncio
async def test_session_event_topic_pattern():
    """Test the session.{id}.event topic pattern used by SessionLoop."""
    bus = EventBus()
    session_id = "abc-123"
    topic = f"session.{session_id}.event"

    queue = bus.subscribe(topic)
    await bus.publish(topic, {"type": "text_chunk", "text": "hello"})
    await bus.publish(topic, {"type": "tool_result", "tool": "read_file"})
    await bus.publish_end(topic)

    events = [e async for e in bus.listen(queue)]
    assert len(events) == 2
    assert events[0]["type"] == "text_chunk"
    assert events[1]["type"] == "tool_result"


@pytest.mark.asyncio
async def test_queue_full_drops_event():
    """When a subscriber's queue is full, events are dropped (not blocking)."""
    bus = EventBus()
    queue = bus.subscribe("test.topic", maxsize=3)

    # Fill the queue (leave room for end sentinel)
    await bus.publish("test.topic", "event-1")
    await bus.publish("test.topic", "event-2")
    await bus.publish("test.topic", "event-3")

    # This should be dropped (queue full), not block
    delivered = await bus.publish("test.topic", "event-4")
    assert delivered == 0  # Dropped

    # Drain one item to make room for end sentinel
    item = await queue.get()
    assert item == "event-1"

    await bus.publish_end("test.topic")
    events = [e async for e in bus.listen(queue)]
    assert events == ["event-2", "event-3"]
