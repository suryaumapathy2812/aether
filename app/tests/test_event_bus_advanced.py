"""Advanced tests for EventBus concurrency and cleanup."""

import asyncio
import pytest
from aether.kernel.event_bus import EventBus


@pytest.mark.asyncio
async def test_event_bus_multiple_subscribers():
    """Multiple subscribers receive the same events independently."""
    bus = EventBus()

    q1 = bus.subscribe("test.topic")
    q2 = bus.subscribe("test.topic")

    assert bus.subscriber_count("test.topic") == 2

    await bus.publish("test.topic", {"msg": "hello"})

    # Both queues should have the event
    assert q1.qsize() == 1
    assert q2.qsize() == 1

    item1 = await q1.get()
    item2 = await q2.get()

    assert item1 == {"msg": "hello"}
    assert item2 == {"msg": "hello"}


@pytest.mark.asyncio
async def test_event_bus_slow_subscriber_dropped_events():
    """If a subscriber's queue is full, events are dropped for that subscriber only."""
    bus = EventBus()

    # q1 has maxsize 1, q2 has maxsize 10
    q1 = bus.subscribe("test.topic", maxsize=1)
    q2 = bus.subscribe("test.topic", maxsize=10)

    # Publish 3 events
    await bus.publish("test.topic", {"msg": "1"})
    await bus.publish("test.topic", {"msg": "2"})
    await bus.publish("test.topic", {"msg": "3"})

    # q1 should only have 1 event (others dropped)
    assert q1.qsize() == 1
    item1 = await q1.get()
    assert item1 == {"msg": "1"}

    # q2 should have all 3 events
    assert q2.qsize() == 3
    items2 = [await q2.get() for _ in range(3)]
    assert items2 == [{"msg": "1"}, {"msg": "2"}, {"msg": "3"}]


@pytest.mark.asyncio
async def test_event_bus_listen_generator():
    """The listen() generator yields events until publish_end is called."""
    bus = EventBus()
    q = bus.subscribe("test.topic")

    async def publisher():
        await bus.publish("test.topic", 1)
        await bus.publish("test.topic", 2)
        await bus.publish_end("test.topic")

    asyncio.create_task(publisher())

    events = [e async for e in bus.listen(q)]
    assert events == [1, 2]


@pytest.mark.asyncio
async def test_event_bus_cleanup():
    """Unsubscribing removes the queue and cleans up empty topics."""
    bus = EventBus()

    q1 = bus.subscribe("test.topic")
    q2 = bus.subscribe("test.topic")

    assert "test.topic" in bus.active_topics()

    bus.unsubscribe("test.topic", q1)
    assert bus.subscriber_count("test.topic") == 1

    bus.unsubscribe("test.topic", q2)
    assert bus.subscriber_count("test.topic") == 0
    assert "test.topic" not in bus.active_topics()
