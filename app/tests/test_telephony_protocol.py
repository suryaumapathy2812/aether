import base64
import json

from aether.voice.telephony_protocol import VobizProtocol, get_protocol


def test_vobiz_parse_start_message() -> None:
    proto = VobizProtocol()
    msg = json.dumps(
        {
            "sequenceNumber": 0,
            "event": "start",
            "start": {
                "callId": "call-123",
                "streamId": "stream-123",
                "mediaFormat": {"encoding": "audio/x-l16", "sampleRate": 16000},
            },
        }
    )

    parsed = proto.parse_media_message(msg)

    assert parsed is not None
    assert parsed["type"] == "start"
    assert parsed["stream_sid"] == "stream-123"
    assert parsed["call_id"] == "call-123"
    assert parsed["content_type"] == "audio/x-l16"
    assert parsed["sample_rate"] == 16000


def test_vobiz_parse_media_message() -> None:
    proto = VobizProtocol()
    payload = b"\x01\x02\x03\x04"
    msg = json.dumps(
        {
            "event": "media",
            "streamId": "stream-123",
            "media": {
                "payload": base64.b64encode(payload).decode("ascii"),
                "contentType": "audio/x-mulaw",
                "sampleRate": 8000,
            },
        }
    )

    parsed = proto.parse_media_message(msg)

    assert parsed is not None
    assert parsed["type"] == "media"
    assert parsed["audio"] == payload
    assert parsed["content_type"] == "audio/x-mulaw"
    assert parsed["sample_rate"] == 8000


def test_vobiz_encode_control_and_audio_messages() -> None:
    proto = VobizProtocol()

    play = json.loads(
        proto.encode_audio_message(
            b"\x00\x01",
            "stream-123",
            content_type="audio/x-l16",
            sample_rate=16000,
        )
    )
    clear = json.loads(proto.create_clear_message("stream-123"))
    checkpoint = json.loads(proto.create_checkpoint_message("stream-123", "cp-1"))

    assert play["event"] == "playAudio"
    assert play["media"]["contentType"] == "audio/x-l16"
    assert play["media"]["sampleRate"] == 16000
    assert clear == {"event": "clearAudio", "streamId": "stream-123"}
    assert checkpoint == {
        "event": "checkpoint",
        "streamId": "stream-123",
        "name": "cp-1",
    }


def test_get_protocol_vobiz_returns_vobiz_protocol() -> None:
    proto = get_protocol("vobiz")
    assert isinstance(proto, VobizProtocol)
