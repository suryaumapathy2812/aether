"""
Aether Frame — the fundamental unit of data flowing through the pipeline.

Every piece of data (audio, text, image, memory context, control signals)
is wrapped in a Frame. Processors consume and produce frames.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FrameType(str, Enum):
    AUDIO = "audio"
    TEXT = "text"
    VISION = "vision"
    MEMORY = "memory"
    CONTROL = "control"
    TOOL_CALL = "tool_call"      # LLM wants to call a tool
    TOOL_RESULT = "tool_result"  # Tool execution result
    STATUS = "status"            # Status/acknowledge for UI/voice


@dataclass
class Frame:
    type: FrameType
    data: Any  # bytes for audio/vision, str for text, dict for memory/control
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


# --- Convenience constructors ---

def audio_frame(data: bytes, sample_rate: int = 16000, channels: int = 1) -> Frame:
    return Frame(
        type=FrameType.AUDIO,
        data=data,
        metadata={"sample_rate": sample_rate, "channels": channels},
    )


def text_frame(text: str, role: str = "user") -> Frame:
    return Frame(
        type=FrameType.TEXT,
        data=text,
        metadata={"role": role},
    )


def vision_frame(image_data: bytes, mime_type: str = "image/jpeg") -> Frame:
    return Frame(
        type=FrameType.VISION,
        data=image_data,
        metadata={"mime_type": mime_type},
    )


def memory_frame(memories: list[str], query: str = "") -> Frame:
    return Frame(
        type=FrameType.MEMORY,
        data={"memories": memories, "query": query},
    )


def control_frame(action: str, **kwargs) -> Frame:
    return Frame(
        type=FrameType.CONTROL,
        data={"action": action, **kwargs},
    )


def tool_call_frame(tool_name: str, arguments: dict, call_id: str = "") -> Frame:
    return Frame(
        type=FrameType.TOOL_CALL,
        data={"tool_name": tool_name, "arguments": arguments, "call_id": call_id},
    )


def tool_result_frame(tool_name: str, output: str, call_id: str = "", error: bool = False) -> Frame:
    return Frame(
        type=FrameType.TOOL_RESULT,
        data={"tool_name": tool_name, "output": output, "call_id": call_id, "error": error},
    )


def status_frame(text: str, tool_name: str = "") -> Frame:
    """Status frame — shown as spinner text in UI, spoken as TTS acknowledge in voice."""
    return Frame(
        type=FrameType.STATUS,
        data={"text": text, "tool_name": tool_name},
    )
