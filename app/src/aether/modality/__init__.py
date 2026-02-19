"""
Modality Adapters — convert transport I/O into kernel jobs.

Each adapter handles a specific modality:
- TextAdapter: text-only I/O (HTTP /chat, WebSocket text mode)
- VoiceAdapter: audio I/O with STT/TTS (WebRTC, WebSocket voice mode)
- SystemAdapter: internal/background jobs (no user I/O)

Adapters are paired with transports via TransportPairing rules.
"""

from aether.modality.base import ModalityAdapter
from aether.modality.system_adapter import SystemAdapter
from aether.modality.text_adapter import TextAdapter
from aether.modality.voice_adapter import VoiceAdapter

__all__ = [
    "ModalityAdapter",
    "TextAdapter",
    "VoiceAdapter",
    "SystemAdapter",
]
