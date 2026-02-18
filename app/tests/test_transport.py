"""Tests for the transport layer — CoreMsg, WebSocketTransport, TransportManager."""

import asyncio
import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aether.transport.core_msg import (
    AudioContent,
    ConnectionState,
    CoreMsg,
    EventContent,
    MsgDirection,
    MsgMetadata,
    TextContent,
)
from aether.transport.base import Transport
from aether.transport.manager import TransportManager
from aether.transport.websocket import WebSocketTransport


# ── CoreMsg Tests ────────────────────────────────────────────────


class TestCoreMsg:
    """Tests for CoreMsg dataclass and factory methods."""

    def test_text_factory(self):
        msg = CoreMsg.text(
            text="hello",
            user_id="u1",
            session_id="s1",
            role="user",
            transport="websocket",
        )
        assert isinstance(msg.content, TextContent)
        assert msg.content.text == "hello"
        assert msg.content.role == "user"
        assert msg.user_id == "u1"
        assert msg.session_id == "s1"
        assert msg.metadata.transport == "websocket"
        assert msg.direction == MsgDirection.INBOUND

    def test_text_factory_assistant_is_outbound(self):
        msg = CoreMsg.text(
            text="hi there",
            user_id="u1",
            session_id="s1",
            role="assistant",
        )
        assert msg.direction == MsgDirection.OUTBOUND

    def test_audio_factory(self):
        audio = b"\x00\x01\x02\x03"
        msg = CoreMsg.audio(
            audio_data=audio,
            user_id="u1",
            session_id="s1",
            sample_rate=24000,
            transport="websocket",
        )
        assert isinstance(msg.content, AudioContent)
        assert msg.content.audio_data == audio
        assert msg.content.sample_rate == 24000
        assert msg.content.channels == 1
        assert msg.direction == MsgDirection.INBOUND

    def test_event_factory(self):
        msg = CoreMsg.event(
            event_type="stream_start",
            user_id="u1",
            session_id="s1",
            payload={"reconnect": True},
            transport="websocket",
        )
        assert isinstance(msg.content, EventContent)
        assert msg.content.event_type == "stream_start"
        assert msg.content.payload == {"reconnect": True}

    def test_event_factory_default_payload(self):
        msg = CoreMsg.event(
            event_type="mute",
            user_id="u1",
            session_id="s1",
        )
        assert msg.content.payload == {}

    def test_unique_ids(self):
        m1 = CoreMsg.text(text="a", user_id="u", session_id="s")
        m2 = CoreMsg.text(text="b", user_id="u", session_id="s")
        assert m1.id != m2.id

    def test_metadata_kwargs_passthrough(self):
        """Extra kwargs to factory methods are passed to MsgMetadata."""
        msg = CoreMsg.text(
            text="hi",
            user_id="u1",
            session_id="s1",
            transport="websocket",
            session_mode="text",
            sentence_index=5,
        )
        assert msg.metadata.session_mode == "text"
        assert msg.metadata.sentence_index == 5


class TestMsgMetadata:
    """Tests for MsgMetadata defaults."""

    def test_defaults(self):
        meta = MsgMetadata()
        assert meta.transport == "unknown"
        assert meta.session_mode == "voice"
        assert meta.sentence_index == 0
        assert isinstance(meta.client_info, dict)
        assert meta.timestamp > 0

    def test_custom_values(self):
        meta = MsgMetadata(
            transport="websocket",
            session_mode="text",
            sentence_index=3,
        )
        assert meta.transport == "websocket"
        assert meta.session_mode == "text"
        assert meta.sentence_index == 3

    def test_default_factory_on_coremsg(self):
        """CoreMsg's field(default_factory=MsgMetadata) should not crash."""
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=TextContent(text="test"),
        )
        assert msg.metadata.transport == "unknown"
        assert msg.metadata.sentence_index == 0


# ── WebSocketTransport Serialization Tests ───────────────────────


class TestWebSocketSerialize:
    """Tests for WebSocketTransport._serialize() — CoreMsg → wire protocol."""

    def setup_method(self):
        self.ws = WebSocketTransport()

    def test_text_chunk(self):
        msg = CoreMsg.text(
            text="Hello world",
            user_id="u1",
            session_id="s1",
            role="assistant",
            transport="text_chunk",
            sentence_index=3,
        )
        result = self.ws._serialize(msg)
        assert result == {"type": "text_chunk", "data": "Hello world", "index": 3}

    def test_text_chunk_default_index(self):
        msg = CoreMsg.text(
            text="Hi",
            user_id="u1",
            session_id="s1",
            role="assistant",
            transport="text_chunk",
        )
        result = self.ws._serialize(msg)
        assert result["index"] == 0

    def test_transcript_final(self):
        msg = CoreMsg.text(
            text="user said this",
            user_id="u1",
            session_id="s1",
            role="system",
            transport="transcript",
        )
        result = self.ws._serialize(msg)
        assert result == {
            "type": "transcript",
            "data": "user said this",
            "interim": False,
        }

    def test_transcript_interim(self):
        msg = CoreMsg.text(
            text="user is say...",
            user_id="u1",
            session_id="s1",
            role="system",
            transport="transcript_interim",
        )
        result = self.ws._serialize(msg)
        assert result == {
            "type": "transcript",
            "data": "user is say...",
            "interim": True,
        }

    def test_status(self):
        msg = CoreMsg.text(
            text="listening...",
            user_id="u1",
            session_id="s1",
            role="system",
            transport="status",
        )
        result = self.ws._serialize(msg)
        assert result == {"type": "status", "data": "listening..."}

    def test_audio_chunk(self):
        audio = b"\x00\x01\x02"
        msg = CoreMsg.audio(
            audio_data=audio,
            user_id="u1",
            session_id="s1",
            transport="audio_chunk",
            sentence_index=2,
        )
        result = self.ws._serialize(msg)
        assert result["type"] == "audio_chunk"
        assert result["data"] == base64.b64encode(audio).decode("utf-8")
        assert result["index"] == 2

    def test_status_audio(self):
        audio = b"\xff\xfe"
        msg = CoreMsg.audio(
            audio_data=audio,
            user_id="u1",
            session_id="s1",
            transport="status_audio",
            sentence_index=5,  # should be overridden to -1
        )
        result = self.ws._serialize(msg)
        assert result["index"] == -1
        assert result["status_audio"] is True

    def test_stream_end(self):
        msg = CoreMsg.event(
            event_type="stream_end",
            user_id="u1",
            session_id="s1",
        )
        result = self.ws._serialize(msg)
        assert result == {"type": "stream_end"}

    def test_tool_result(self):
        msg = CoreMsg.event(
            event_type="tool_result",
            user_id="u1",
            session_id="s1",
            payload={"name": "read_file", "output": "contents"},
        )
        result = self.ws._serialize(msg)
        assert result["type"] == "tool_result"
        assert json.loads(result["data"]) == {"name": "read_file", "output": "contents"}

    def test_notification(self):
        msg = CoreMsg.event(
            event_type="notification",
            user_id="u1",
            session_id="s1",
            payload={"level": "nudge", "text": "New email"},
        )
        result = self.ws._serialize(msg)
        assert result["type"] == "notification"

    def test_ready_event(self):
        msg = CoreMsg.event(
            event_type="ready",
            user_id="u1",
            session_id="s1",
        )
        result = self.ws._serialize(msg)
        assert result == {"type": "status", "data": "listening..."}

    def test_unknown_event_returns_none(self):
        msg = CoreMsg.event(
            event_type="some_unknown_event",
            user_id="u1",
            session_id="s1",
        )
        result = self.ws._serialize(msg)
        assert result is None


# ── WebSocketTransport Connection State Tests ────────────────────


class TestWebSocketTransportState:
    """Tests for WebSocketTransport connection tracking."""

    def setup_method(self):
        self.ws = WebSocketTransport()

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        await self.ws.start()
        assert self.ws.is_running is True

    @pytest.mark.asyncio
    async def test_stop_clears_state(self):
        await self.ws.start()
        await self.ws.stop()
        assert self.ws.is_running is False
        assert len(self.ws._connections) == 0

    @pytest.mark.asyncio
    async def test_is_connected_false_by_default(self):
        assert await self.ws.is_connected("nobody") is False

    @pytest.mark.asyncio
    async def test_get_connected_users_empty(self):
        assert await self.ws.get_connected_users() == []

    @pytest.mark.asyncio
    async def test_get_status(self):
        await self.ws.start()
        status = await self.ws.get_status()
        assert status["transport"] == "websocket"
        assert status["connections"] == 0
        assert status["users"] == 0


# ── TransportManager Tests ───────────────────────────────────────


class _FakeTransport(Transport):
    """Minimal transport for testing the manager."""

    name = "fake"

    def __init__(self):
        super().__init__()
        self._users: dict[str, bool] = {}
        self.sent: list[tuple[str, CoreMsg]] = []
        self.broadcast_msgs: list[CoreMsg] = []

    async def start(self):
        self._running = True

    async def stop(self):
        self._running = False

    async def send(self, user_id: str, msg: CoreMsg) -> None:
        self.sent.append((user_id, msg))

    async def broadcast(self, msg: CoreMsg) -> None:
        self.broadcast_msgs.append(msg)

    async def get_connected_users(self) -> list[str]:
        return list(self._users.keys())

    async def is_connected(self, user_id: str) -> bool:
        return user_id in self._users

    async def get_status(self) -> dict:
        return {"transport": self.name, "connections": len(self._users)}

    def simulate_connect(self, user_id: str):
        self._users[user_id] = True

    def simulate_disconnect(self, user_id: str):
        self._users.pop(user_id, None)


class TestTransportManager:
    """Tests for TransportManager routing and connection tracking."""

    def _make_core(self):
        core = AsyncMock()
        core.start = AsyncMock()
        core.stop = AsyncMock()
        core.health_check = AsyncMock(return_value={"status": "ok"})
        core.process_message = AsyncMock(return_value=iter([]))
        return core

    @pytest.mark.asyncio
    async def test_register_transport(self):
        core = self._make_core()
        mgr = TransportManager(core)
        transport = _FakeTransport()
        await mgr.register_transport(transport)
        assert "fake" in mgr.transports
        assert mgr.get_transport("fake") is transport

    @pytest.mark.asyncio
    async def test_start_all_starts_core_and_transports(self):
        core = self._make_core()
        mgr = TransportManager(core)
        transport = _FakeTransport()
        await mgr.register_transport(transport)
        await mgr.start_all()
        core.start.assert_awaited_once()
        assert transport.is_running

    @pytest.mark.asyncio
    async def test_stop_all(self):
        core = self._make_core()
        mgr = TransportManager(core)
        transport = _FakeTransport()
        await mgr.register_transport(transport)
        await mgr.start_all()
        await mgr.stop_all()
        core.stop.assert_awaited_once()
        assert not transport.is_running

    @pytest.mark.asyncio
    async def test_connection_tracking_connect(self):
        core = self._make_core()
        mgr = TransportManager(core)
        transport = _FakeTransport()
        await mgr.register_transport(transport)

        # Simulate connection change with transport name
        await mgr._handle_connection_change("user1", ConnectionState.CONNECTED, "fake")
        assert await mgr.is_user_connected("user1")
        users = await mgr.get_connected_users()
        assert "user1" in users

    @pytest.mark.asyncio
    async def test_connection_tracking_disconnect(self):
        core = self._make_core()
        mgr = TransportManager(core)
        transport = _FakeTransport()
        await mgr.register_transport(transport)

        await mgr._handle_connection_change("user1", ConnectionState.CONNECTED, "fake")
        await mgr._handle_connection_change(
            "user1", ConnectionState.DISCONNECTED, "fake"
        )
        assert not await mgr.is_user_connected("user1")

    @pytest.mark.asyncio
    async def test_connection_tracking_multi_transport(self):
        """User connected via two transports — disconnect from one keeps them connected."""
        core = self._make_core()
        mgr = TransportManager(core)
        t1 = _FakeTransport()
        t1.name = "ws"
        t2 = _FakeTransport()
        t2.name = "webrtc"
        await mgr.register_transport(t1)
        await mgr.register_transport(t2)

        await mgr._handle_connection_change("user1", ConnectionState.CONNECTED, "ws")
        await mgr._handle_connection_change(
            "user1", ConnectionState.CONNECTED, "webrtc"
        )
        assert await mgr.is_user_connected("user1")

        # Disconnect from one transport
        await mgr._handle_connection_change("user1", ConnectionState.DISCONNECTED, "ws")
        # Still connected via the other
        assert await mgr.is_user_connected("user1")

        # Disconnect from the other
        await mgr._handle_connection_change(
            "user1", ConnectionState.DISCONNECTED, "webrtc"
        )
        assert not await mgr.is_user_connected("user1")

    @pytest.mark.asyncio
    async def test_send_to_user_routes_to_transport(self):
        core = self._make_core()
        mgr = TransportManager(core)
        transport = _FakeTransport()
        await mgr.register_transport(transport)

        await mgr._handle_connection_change("user1", ConnectionState.CONNECTED, "fake")

        msg = CoreMsg.text(text="hello", user_id="user1", session_id="s1")
        await mgr.send_to_user("user1", msg)

        assert len(transport.sent) == 1
        assert transport.sent[0][0] == "user1"

    @pytest.mark.asyncio
    async def test_send_to_disconnected_user_is_noop(self):
        core = self._make_core()
        mgr = TransportManager(core)
        transport = _FakeTransport()
        await mgr.register_transport(transport)

        msg = CoreMsg.text(text="hello", user_id="nobody", session_id="s1")
        await mgr.send_to_user("nobody", msg)
        assert len(transport.sent) == 0

    @pytest.mark.asyncio
    async def test_broadcast(self):
        core = self._make_core()
        mgr = TransportManager(core)
        transport = _FakeTransport()
        await mgr.register_transport(transport)

        await mgr._handle_connection_change("u1", ConnectionState.CONNECTED, "fake")
        await mgr._handle_connection_change("u2", ConnectionState.CONNECTED, "fake")

        msg = CoreMsg.event(
            event_type="notification",
            user_id="",
            session_id="",
            payload={"text": "broadcast"},
        )
        await mgr.broadcast(msg)

        # Should have sent to both users
        assert len(transport.sent) == 2
        sent_users = {s[0] for s in transport.sent}
        assert sent_users == {"u1", "u2"}

    @pytest.mark.asyncio
    async def test_get_status(self):
        core = self._make_core()
        mgr = TransportManager(core)
        transport = _FakeTransport()
        await mgr.register_transport(transport)

        status = await mgr.get_status()
        assert "core" in status
        assert "transports" in status
        assert "fake" in status["transports"]
        assert "connections" in status

    @pytest.mark.asyncio
    async def test_reconnect_updates_transport(self):
        core = self._make_core()
        mgr = TransportManager(core)
        transport = _FakeTransport()
        await mgr.register_transport(transport)

        await mgr._handle_connection_change(
            "user1", ConnectionState.RECONNECTED, "fake"
        )
        assert await mgr.is_user_connected("user1")
