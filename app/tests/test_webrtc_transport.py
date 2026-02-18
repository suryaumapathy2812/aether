"""Tests for the SmallWebRTC transport layer.

Tests cover:
- RawAudioTrack audio chunking and silence generation
- SmallWebRTCTransport state management (start/stop/status)
- Data channel message serialization (CoreMsg → client JSON)
- Data channel message normalization (client JSON → CoreMsg)
- Session lifecycle (connect/disconnect tracking)
- Graceful degradation when aiortc is not installed

Note: These tests mock aiortc internals so they run without
actually creating WebRTC peer connections.
"""

import asyncio
import base64
import json
import time
from dataclasses import dataclass
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


# ── Check if aiortc is available ─────────────────────────────

try:
    from aether.transport.webrtc import (
        AIORTC_AVAILABLE,
        RawAudioTrack,
        SmallWebRTCTransport,
        WebRTCSession,
    )

    HAS_WEBRTC = AIORTC_AVAILABLE
except (ImportError, RuntimeError):
    HAS_WEBRTC = False

# Skip all tests if aiortc is not installed
pytestmark = pytest.mark.skipif(
    not HAS_WEBRTC,
    reason="aiortc not installed — WebRTC tests skipped",
)


# ── RawAudioTrack Tests ─────────────────────────────────────────


class TestRawAudioTrack:
    """Tests for the RawAudioTrack (outbound audio to WebRTC client)."""

    def test_init_defaults(self):
        track = RawAudioTrack(sample_rate=24000)
        assert track._sample_rate == 24000
        assert track._samples_per_10ms == 240  # 24000 * 10 / 1000
        assert track._bytes_per_10ms == 480  # 240 * 2 (16-bit)
        assert track._timestamp == 0
        assert len(track._chunk_queue) == 0

    def test_init_16khz(self):
        track = RawAudioTrack(sample_rate=16000)
        assert track._samples_per_10ms == 160
        assert track._bytes_per_10ms == 320

    def test_add_audio_exact_10ms(self):
        """Adding exactly 10ms of audio creates one chunk."""
        track = RawAudioTrack(sample_rate=24000)
        audio = bytes(480)  # exactly 10ms at 24kHz
        track.add_audio_bytes(audio)
        assert len(track._chunk_queue) == 1

    def test_add_audio_multiple_chunks(self):
        """Adding 30ms of audio creates three 10ms chunks."""
        track = RawAudioTrack(sample_rate=24000)
        audio = bytes(480 * 3)  # 30ms
        track.add_audio_bytes(audio)
        assert len(track._chunk_queue) == 3

    def test_add_audio_pads_partial_chunk(self):
        """Partial last chunk is zero-padded to 10ms."""
        track = RawAudioTrack(sample_rate=24000)
        audio = bytes(500)  # 480 + 20 extra bytes
        track.add_audio_bytes(audio)
        assert len(track._chunk_queue) == 2
        # Second chunk should be padded to 480 bytes
        assert len(track._chunk_queue[1]) == 480

    @pytest.mark.asyncio
    async def test_recv_silence_when_empty(self):
        """recv() returns silence when queue is empty."""
        track = RawAudioTrack(sample_rate=24000)
        frame = await track.recv()
        assert frame.sample_rate == 24000
        assert frame.pts == 0
        # Silence = all zeros
        import numpy as np

        samples = frame.to_ndarray()
        assert np.all(samples == 0)

    @pytest.mark.asyncio
    async def test_recv_returns_queued_audio(self):
        """recv() returns queued audio data."""
        track = RawAudioTrack(sample_rate=24000)
        # Create non-zero audio
        import numpy as np

        samples = np.ones(240, dtype=np.int16)
        audio_bytes = samples.tobytes()
        track.add_audio_bytes(audio_bytes)

        frame = await track.recv()
        result = frame.to_ndarray().flatten()
        assert np.all(result == 1)

    @pytest.mark.asyncio
    async def test_recv_increments_timestamp(self):
        """Each recv() advances the timestamp by samples_per_10ms."""
        track = RawAudioTrack(sample_rate=24000)
        frame1 = await track.recv()
        assert frame1.pts == 0
        frame2 = await track.recv()
        assert frame2.pts == 240  # samples_per_10ms


# ── SmallWebRTCTransport State Tests ─────────────────────────────


class TestSmallWebRTCTransportState:
    """Tests for transport lifecycle and state management."""

    def test_transport_name(self):
        transport = SmallWebRTCTransport()
        assert transport.name == "webrtc"

    def test_is_transport_subclass(self):
        transport = SmallWebRTCTransport()
        assert isinstance(transport, Transport)

    @pytest.mark.asyncio
    async def test_start_stop(self):
        transport = SmallWebRTCTransport()
        await transport.start()
        assert transport.is_running
        await transport.stop()
        assert not transport.is_running

    @pytest.mark.asyncio
    async def test_get_status(self):
        transport = SmallWebRTCTransport()
        await transport.start()
        status = await transport.get_status()
        assert status["transport"] == "webrtc"
        assert status["connections"] == 0
        assert status["users"] == 0
        assert status["aiortc_available"] is True
        await transport.stop()

    @pytest.mark.asyncio
    async def test_no_connected_users_initially(self):
        transport = SmallWebRTCTransport()
        users = await transport.get_connected_users()
        assert users == []

    @pytest.mark.asyncio
    async def test_is_connected_false_initially(self):
        transport = SmallWebRTCTransport()
        assert not await transport.is_connected("user1")

    @pytest.mark.asyncio
    async def test_send_to_unknown_user_is_noop(self):
        """Sending to a user with no session should not raise."""
        transport = SmallWebRTCTransport()
        msg = CoreMsg.text(
            text="hello",
            user_id="nobody",
            session_id="s1",
            transport="webrtc",
        )
        await transport.send("nobody", msg)  # should not raise

    def test_ice_server_config(self):
        transport = SmallWebRTCTransport(
            ice_servers=[{"urls": "stun:stun.example.com:3478"}]
        )
        assert len(transport._ice_servers) == 1

    def test_no_ice_servers(self):
        transport = SmallWebRTCTransport(ice_servers=None)
        assert transport._ice_servers == []

    def test_repr(self):
        transport = SmallWebRTCTransport()
        assert "SmallWebRTCTransport" in repr(transport)
        assert "connections=0" in repr(transport)


# ── Data Channel Serialization Tests ─────────────────────────────


class TestDataChannelSerialize:
    """Tests for _serialize_for_data_channel (CoreMsg → client JSON)."""

    def _make_transport(self):
        return SmallWebRTCTransport()

    def test_text_chunk(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=TextContent(text="Hello world", role="assistant"),
            metadata=MsgMetadata(transport="text_chunk", sentence_index=2),
        )
        result = t._serialize_for_data_channel(msg)
        assert result == {"type": "text_chunk", "data": "Hello world", "index": 2}

    def test_assistant_text_is_text_chunk(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=TextContent(text="response", role="assistant"),
            metadata=MsgMetadata(transport="whatever"),
        )
        result = t._serialize_for_data_channel(msg)
        assert result["type"] == "text_chunk"

    def test_transcript(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=TextContent(text="hello", role="user"),
            metadata=MsgMetadata(transport="transcript"),
        )
        result = t._serialize_for_data_channel(msg)
        assert result == {"type": "transcript", "data": "hello", "interim": False}

    def test_transcript_interim(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=TextContent(text="hel", role="user"),
            metadata=MsgMetadata(transport="transcript_interim"),
        )
        result = t._serialize_for_data_channel(msg)
        assert result == {"type": "transcript", "data": "hel", "interim": True}

    def test_status(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=TextContent(text="thinking...", role="user"),
            metadata=MsgMetadata(transport="status"),
        )
        result = t._serialize_for_data_channel(msg)
        assert result == {"type": "status", "data": "thinking..."}

    def test_stream_end(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=EventContent(event_type="stream_end"),
            metadata=MsgMetadata(transport="webrtc"),
        )
        result = t._serialize_for_data_channel(msg)
        assert result == {"type": "stream_end"}

    def test_tool_result(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=EventContent(
                event_type="tool_result",
                payload={"result": "done"},
            ),
            metadata=MsgMetadata(transport="webrtc"),
        )
        result = t._serialize_for_data_channel(msg)
        assert result["type"] == "tool_result"
        assert json.loads(result["data"]) == {"result": "done"}

    def test_notification(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=EventContent(
                event_type="notification",
                payload={"text": "New email"},
            ),
            metadata=MsgMetadata(transport="notification"),
        )
        result = t._serialize_for_data_channel(msg)
        assert result["type"] == "notification"

    def test_ready_event(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=EventContent(event_type="ready"),
            metadata=MsgMetadata(transport="webrtc"),
        )
        result = t._serialize_for_data_channel(msg)
        assert result == {"type": "status", "data": "listening..."}

    def test_unknown_event_returns_none(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=EventContent(event_type="some_internal_event"),
            metadata=MsgMetadata(transport="webrtc"),
        )
        result = t._serialize_for_data_channel(msg)
        assert result is None

    def test_audio_content_base64(self):
        """AudioContent in data channel serialization (fallback path)."""
        t = self._make_transport()
        audio = b"\x01\x02\x03\x04"
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=AudioContent(audio_data=audio),
            metadata=MsgMetadata(transport="audio_chunk", sentence_index=1),
        )
        result = t._serialize_for_data_channel(msg)
        assert result["type"] == "audio_chunk"
        assert base64.b64decode(result["data"]) == audio
        assert result["index"] == 1

    def test_status_audio_index_minus_one(self):
        t = self._make_transport()
        msg = CoreMsg(
            user_id="u1",
            session_id="s1",
            content=AudioContent(audio_data=b"\x00"),
            metadata=MsgMetadata(transport="status_audio"),
        )
        result = t._serialize_for_data_channel(msg)
        assert result["index"] == -1
        assert result["status_audio"] is True


# ── Data Channel Normalization Tests ─────────────────────────────


class TestDataChannelNormalize:
    """Tests for _handle_data_channel_message (client JSON → CoreMsg)."""

    @pytest.mark.asyncio
    async def test_text_message(self):
        """Text message from data channel → CoreMsg.text()."""
        transport = SmallWebRTCTransport()
        received = []
        transport.on_message(AsyncMock(side_effect=lambda msg: received.append(msg)))

        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
        )

        await transport._handle_data_channel_message(
            session, json.dumps({"type": "text", "data": "hello"})
        )

        assert len(received) == 1
        msg = received[0]
        assert isinstance(msg.content, TextContent)
        assert msg.content.text == "hello"
        assert msg.user_id == "user1"
        assert msg.metadata.transport == "webrtc"

    @pytest.mark.asyncio
    async def test_mute_message(self):
        transport = SmallWebRTCTransport()
        received = []
        transport.on_message(AsyncMock(side_effect=lambda msg: received.append(msg)))

        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
        )

        await transport._handle_data_channel_message(
            session, json.dumps({"type": "mute"})
        )

        assert len(received) == 1
        assert isinstance(received[0].content, EventContent)
        assert received[0].content.event_type == "mute"

    @pytest.mark.asyncio
    async def test_unmute_message(self):
        transport = SmallWebRTCTransport()
        received = []
        transport.on_message(AsyncMock(side_effect=lambda msg: received.append(msg)))

        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
        )

        await transport._handle_data_channel_message(
            session, json.dumps({"type": "unmute"})
        )

        assert len(received) == 1
        assert received[0].content.event_type == "unmute"

    @pytest.mark.asyncio
    async def test_session_config_message(self):
        transport = SmallWebRTCTransport()
        received = []
        transport.on_message(AsyncMock(side_effect=lambda msg: received.append(msg)))

        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
        )

        await transport._handle_data_channel_message(
            session, json.dumps({"type": "session_config", "mode": "text"})
        )

        assert len(received) == 1
        assert received[0].content.event_type == "session_config"
        assert received[0].content.payload == {"mode": "text"}
        assert session.mode == "text"

    @pytest.mark.asyncio
    async def test_ping_pong(self):
        """Ping message updates last_ping_time and sends pong."""
        transport = SmallWebRTCTransport()
        transport.on_message(AsyncMock())

        dc = MagicMock()
        dc.readyState = "open"
        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
            data_channel=dc,
        )

        before = session.last_ping_time
        await transport._handle_data_channel_message(session, "ping")
        assert session.last_ping_time >= before
        dc.send.assert_called_once_with("pong")

    @pytest.mark.asyncio
    async def test_invalid_json_ignored(self):
        """Invalid JSON doesn't raise, just logs warning."""
        transport = SmallWebRTCTransport()
        transport.on_message(AsyncMock())

        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
        )

        # Should not raise
        await transport._handle_data_channel_message(session, "not json {{{")

    @pytest.mark.asyncio
    async def test_image_message(self):
        transport = SmallWebRTCTransport()
        received = []
        transport.on_message(AsyncMock(side_effect=lambda msg: received.append(msg)))

        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
        )

        await transport._handle_data_channel_message(
            session,
            json.dumps(
                {
                    "type": "image",
                    "data": "base64data",
                    "mime": "image/png",
                }
            ),
        )

        assert len(received) == 1
        assert received[0].content.event_type == "image"
        assert received[0].content.payload["mime"] == "image/png"

    @pytest.mark.asyncio
    async def test_notification_feedback(self):
        transport = SmallWebRTCTransport()
        received = []
        transport.on_message(AsyncMock(side_effect=lambda msg: received.append(msg)))

        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
        )

        await transport._handle_data_channel_message(
            session,
            json.dumps(
                {
                    "type": "notification_feedback",
                    "data": {"event_id": "e1", "action": "dismiss"},
                }
            ),
        )

        assert len(received) == 1
        assert received[0].content.event_type == "notification_feedback"


# ── Session Outbound Tests ───────────────────────────────────────


class TestSessionOutbound:
    """Tests for _send_to_session (CoreMsg → WebRTC output)."""

    @pytest.mark.asyncio
    async def test_audio_content_pushed_to_track(self):
        """AudioContent is pushed to the RawAudioTrack."""
        transport = SmallWebRTCTransport()
        audio_track = MagicMock()
        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
            audio_track=audio_track,
        )

        audio_data = bytes(480)
        msg = CoreMsg(
            user_id="user1",
            session_id="pc-test",
            content=AudioContent(audio_data=audio_data),
            metadata=MsgMetadata(transport="webrtc"),
        )

        await transport._send_to_session(session, msg)
        audio_track.add_audio_bytes.assert_called_once_with(audio_data)

    @pytest.mark.asyncio
    async def test_text_content_sent_via_data_channel(self):
        """TextContent is serialized and sent via data channel."""
        transport = SmallWebRTCTransport()
        dc = MagicMock()
        dc.readyState = "open"
        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
            data_channel=dc,
        )

        msg = CoreMsg(
            user_id="user1",
            session_id="pc-test",
            content=TextContent(text="Hello", role="assistant"),
            metadata=MsgMetadata(transport="text_chunk", sentence_index=0),
        )

        await transport._send_to_session(session, msg)
        dc.send.assert_called_once()
        sent = json.loads(dc.send.call_args[0][0])
        assert sent == {"type": "text_chunk", "data": "Hello", "index": 0}

    @pytest.mark.asyncio
    async def test_no_data_channel_is_noop(self):
        """If data channel is None, text send is a no-op."""
        transport = SmallWebRTCTransport()
        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
            data_channel=None,
        )

        msg = CoreMsg(
            user_id="user1",
            session_id="pc-test",
            content=TextContent(text="Hello", role="assistant"),
            metadata=MsgMetadata(transport="text_chunk"),
        )

        # Should not raise
        await transport._send_to_session(session, msg)

    @pytest.mark.asyncio
    async def test_closed_data_channel_is_noop(self):
        """If data channel is not open, send is skipped."""
        transport = SmallWebRTCTransport()
        dc = MagicMock()
        dc.readyState = "closed"
        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
            data_channel=dc,
        )

        msg = CoreMsg(
            user_id="user1",
            session_id="pc-test",
            content=TextContent(text="Hello", role="assistant"),
            metadata=MsgMetadata(transport="text_chunk"),
        )

        await transport._send_to_session(session, msg)
        dc.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_event_content_sent_via_data_channel(self):
        """EventContent is serialized and sent via data channel."""
        transport = SmallWebRTCTransport()
        dc = MagicMock()
        dc.readyState = "open"
        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
            data_channel=dc,
        )

        msg = CoreMsg(
            user_id="user1",
            session_id="pc-test",
            content=EventContent(event_type="stream_end"),
            metadata=MsgMetadata(transport="webrtc"),
        )

        await transport._send_to_session(session, msg)
        dc.send.assert_called_once()
        sent = json.loads(dc.send.call_args[0][0])
        assert sent == {"type": "stream_end"}


# ── Session Lifecycle Tests ──────────────────────────────────────


class TestSessionLifecycle:
    """Tests for session tracking and disconnect handling."""

    @pytest.mark.asyncio
    async def test_handle_disconnect_cleans_up(self):
        """Disconnect removes session from tracking dicts."""
        transport = SmallWebRTCTransport()
        transport.on_message(AsyncMock())
        transport.on_connection_change(AsyncMock())

        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
            connected=True,
        )
        transport._sessions["pc-test"] = session
        transport._user_sessions["user1"] = "pc-test"

        await transport._handle_disconnect(session)

        assert "pc-test" not in transport._sessions
        assert "user1" not in transport._user_sessions

    @pytest.mark.asyncio
    async def test_handle_disconnect_sends_disconnect_event(self):
        """Disconnect sends a disconnect CoreMsg to the core."""
        transport = SmallWebRTCTransport()
        received = []
        transport.on_message(AsyncMock(side_effect=lambda msg: received.append(msg)))
        transport.on_connection_change(AsyncMock())

        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
            connected=True,
        )
        transport._sessions["pc-test"] = session
        transport._user_sessions["user1"] = "pc-test"

        await transport._handle_disconnect(session)

        assert len(received) == 1
        assert isinstance(received[0].content, EventContent)
        assert received[0].content.event_type == "disconnect"
        assert received[0].user_id == "user1"

    @pytest.mark.asyncio
    async def test_handle_disconnect_notifies_connection_change(self):
        """Disconnect notifies the connection callback."""
        transport = SmallWebRTCTransport()
        transport.on_message(AsyncMock())
        conn_callback = AsyncMock()
        transport.on_connection_change(conn_callback)

        session = WebRTCSession(
            pc_id="pc-test",
            pc=MagicMock(),
            user_id="user1",
            connected=True,
        )
        transport._sessions["pc-test"] = session
        transport._user_sessions["user1"] = "pc-test"

        await transport._handle_disconnect(session)

        conn_callback.assert_called_once_with(
            "user1", ConnectionState.DISCONNECTED, "webrtc"
        )

    @pytest.mark.asyncio
    async def test_close_session_cancels_audio_task(self):
        """Closing a session cancels the audio read task."""
        transport = SmallWebRTCTransport()

        # Create a real task that we can cancel
        async def fake_audio_loop():
            await asyncio.sleep(999)

        mock_task = asyncio.create_task(fake_audio_loop())
        mock_pc = AsyncMock()

        session = WebRTCSession(
            pc_id="pc-test",
            pc=mock_pc,
            user_id="user1",
            audio_read_task=mock_task,
        )
        transport._sessions["pc-test"] = session
        transport._user_sessions["user1"] = "pc-test"

        await transport._close_session(session)

        assert mock_task.cancelled()
        mock_pc.close.assert_called_once()
        assert "pc-test" not in transport._sessions

    @pytest.mark.asyncio
    async def test_stop_closes_all_sessions(self):
        """stop() closes all peer connections."""
        transport = SmallWebRTCTransport()
        await transport.start()

        mock_pc = AsyncMock()
        session = WebRTCSession(
            pc_id="pc-test",
            pc=mock_pc,
            user_id="user1",
        )
        transport._sessions["pc-test"] = session
        transport._user_sessions["user1"] = "pc-test"

        await transport.stop()

        mock_pc.close.assert_called_once()
        assert len(transport._sessions) == 0
        assert len(transport._user_sessions) == 0
        assert not transport.is_running


# ── Broadcast Tests ──────────────────────────────────────────────


class TestBroadcast:
    """Tests for broadcast to all sessions."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self):
        """Broadcast sends to all connected sessions."""
        transport = SmallWebRTCTransport()

        dc1 = MagicMock()
        dc1.readyState = "open"
        dc2 = MagicMock()
        dc2.readyState = "open"

        transport._sessions["pc1"] = WebRTCSession(
            pc_id="pc1", pc=MagicMock(), user_id="u1", data_channel=dc1
        )
        transport._sessions["pc2"] = WebRTCSession(
            pc_id="pc2", pc=MagicMock(), user_id="u2", data_channel=dc2
        )
        transport._user_sessions["u1"] = "pc1"
        transport._user_sessions["u2"] = "pc2"

        msg = CoreMsg(
            user_id="",
            session_id="",
            content=EventContent(event_type="stream_end"),
            metadata=MsgMetadata(transport="webrtc"),
        )

        await transport.broadcast(msg)

        dc1.send.assert_called_once()
        dc2.send.assert_called_once()
