from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aether.ws.sidecar import WSSidecar


class _FakeAgent:
    def __init__(self) -> None:
        self._memory_store = AsyncMock()

    def subscribe_notifications(self, callback):
        return None

    def unsubscribe_notifications(self, callback):
        return None


@pytest.mark.asyncio
async def test_feedback_dismissed_syncs_notification_status() -> None:
    agent = _FakeAgent()
    sidecar = WSSidecar(agent=agent)
    sidecar.push = AsyncMock()

    await sidecar._handle_feedback(
        {
            "action": "dismissed",
            "plugin": "gmail",
            "sender": "alerts@example.com",
            "notification_id": "12",
            "device_id": "dev-1",
        }
    )

    agent._memory_store.mark_dismissed.assert_awaited_once_with(12)
    sidecar.push.assert_awaited_once()
    event_type, payload = sidecar.push.await_args.args
    assert event_type == "notification_status"
    assert payload["notification_id"] == 12
    assert payload["status"] == "dismissed"
    assert payload["device_id"] == "dev-1"


@pytest.mark.asyncio
async def test_feedback_engaged_marks_delivered_and_syncs() -> None:
    agent = _FakeAgent()
    sidecar = WSSidecar(agent=agent)
    sidecar.push = AsyncMock()

    await sidecar._handle_feedback(
        {
            "action": "engaged",
            "plugin": "calendar",
            "sender": "calendar.google.com",
            "notification_id": 7,
        }
    )

    agent._memory_store.mark_delivered.assert_awaited_once_with(7)
    agent._memory_store._store_fact.assert_awaited_once()
    event_type, payload = sidecar.push.await_args.args
    assert event_type == "notification_status"
    assert payload["status"] == "delivered"
    assert payload["notification_id"] == 7
