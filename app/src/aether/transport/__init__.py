"""
Aether Transport Layer

A modular transport system that handles all client connections:
- WebSocket (text and voice)
- WebRTC (self-hosted via aiortc — SmallWebRTC)
- Push (notifications)
- And more...

The transport layer is a facade that:
- Accepts connections from various client types
- Normalizes messages to CoreMsg format
- Routes messages to/from the Aether Core (via KernelCore)
- Manages connection lifecycle

Usage:
    from aether.transport import TransportManager, CoreHandler, WebSocketTransport

    # Create core handler with your components
    core = CoreHandler(
        llm_provider=...,
        memory_store=...,
        tool_registry=...,
        ...
    )

    # Create transport manager
    manager = TransportManager(core)

    # Register transports
    manager.register_transport(WebSocketTransport())

    # Optional: register WebRTC transport (requires aiortc)
    try:
        from aether.transport.webrtc import SmallWebRTCTransport
        manager.register_transport(SmallWebRTCTransport())
    except RuntimeError:
        pass  # aiortc not installed

    # Start all
    await manager.start_all()
"""

from aether.transport.base import Transport
from aether.transport.core_msg import (
    AudioContent,
    ConnectionState,
    CoreMsg,
    EventContent,
    MsgDirection,
    MsgMetadata,
    TextContent,
)
from aether.transport.interface import CoreInterface
from aether.transport.manager import TransportManager
from aether.transport.websocket import WebSocketTransport


def __getattr__(name: str):
    """Lazy import for backward compat — avoids circular import with kernel.core."""
    if name == "CoreHandler":
        from aether.kernel.core import KernelCore

        return KernelCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Optional: WebRTC transport (requires aiortc)
try:
    from aether.transport.webrtc import SmallWebRTCTransport

    _HAS_WEBRTC = True
except (ImportError, RuntimeError):
    _HAS_WEBRTC = False

__all__ = [
    # Core interfaces
    "CoreInterface",
    "CoreHandler",  # backward compat alias for KernelCore
    "TransportManager",
    "Transport",
    # Message types
    "CoreMsg",
    "MsgDirection",
    "MsgMetadata",
    "ConnectionState",
    "TextContent",
    "AudioContent",
    "EventContent",
    # Transports
    "WebSocketTransport",
]

if _HAS_WEBRTC:
    __all__.append("SmallWebRTCTransport")
