"""Tests for the Frame system — the foundation of everything."""

import pytest
from aether.core.frames import (
    Frame,
    FrameType,
    audio_frame,
    text_frame,
    vision_frame,
    memory_frame,
    control_frame,
)


def test_frame_types_are_strings():
    """FrameType values should be usable as strings."""
    assert FrameType.AUDIO == "audio"
    assert FrameType.TEXT == "text"
    assert FrameType.VISION == "vision"
    assert FrameType.MEMORY == "memory"
    assert FrameType.CONTROL == "control"


def test_audio_frame():
    f = audio_frame(b"\x00\x01\x02", sample_rate=16000)
    assert f.type == FrameType.AUDIO
    assert f.data == b"\x00\x01\x02"
    assert f.metadata["sample_rate"] == 16000
    assert f.metadata["channels"] == 1
    assert f.id  # Should have a UUID
    assert f.timestamp > 0


def test_text_frame():
    f = text_frame("hello world", role="user")
    assert f.type == FrameType.TEXT
    assert f.data == "hello world"
    assert f.metadata["role"] == "user"


def test_text_frame_assistant():
    f = text_frame("response", role="assistant")
    assert f.metadata["role"] == "assistant"


def test_vision_frame():
    f = vision_frame(b"\xff\xd8", mime_type="image/jpeg")
    assert f.type == FrameType.VISION
    assert f.data == b"\xff\xd8"
    assert f.metadata["mime_type"] == "image/jpeg"


def test_memory_frame():
    f = memory_frame(["fact1", "fact2"], query="test query")
    assert f.type == FrameType.MEMORY
    assert f.data["memories"] == ["fact1", "fact2"]
    assert f.data["query"] == "test query"


def test_control_frame():
    f = control_frame("utterance_end", transcript="hello")
    assert f.type == FrameType.CONTROL
    assert f.data["action"] == "utterance_end"
    assert f.data["transcript"] == "hello"


def test_frame_unique_ids():
    """Every frame should get a unique ID."""
    f1 = text_frame("a")
    f2 = text_frame("b")
    assert f1.id != f2.id
