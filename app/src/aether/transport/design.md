"""
Transport Layer Design Document
===============================

## Overview

The transport layer is a facade that handles all client connections (WebSocket, WebRTC, HTTP push)
and routes messages between clients and the Aether Core. The Core (LLMProcessor, Memory, Tools,
Plugins) remains unchanged - it only speaks in Frames.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRANSPORT LAYER                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 TransportManager                           │  │
│  │  • Registers transports                                   │  │
│  │  • Routes incoming messages to Core                       │  │
│  │  • Routes outgoing messages to correct transport         │  │
│  │  • Manages connection lifecycle                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│    ┌─────────────┬───────────┼───────────┬─────────────┐        │
│    ▼             ▼           ▼           ▼             ▼        │
│ ┌──────┐   ┌────────┐  ┌────────┐  ┌────────┐   ┌──────┐   │
│ │  WS  │   │ Daily  │  │LiveKit │  │ Twilio │   │ REST │   │
│ │Transp│   │Transp  │  │Transp  │  │Transp  │   │Notif │   │
│ └──────┘   └────────┘  └────────┘  └────────┘   └──────┘   │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼ CoreMsg (unified)
┌─────────────────────────────────────────────────────────────────┐
│                     AETHER CORE                                 │
│  • LLMProcessor (agentic loop)                                  │
│  • Memory (4-tier)                                             │
│  • Tools, Plugins, Skills                                      │
│  • Providers (STT, LLM, TTS)                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Interface

The CoreInterface is the adapter between the transport layer and the Aether Core.
It translates CoreMsg → Frames and Frames → CoreMsg.

### CoreMsg (Unified Message Format)

```python
@dataclass
class CoreMsg:
    """Unified message format between transport and core."""
    
    # Identity
    user_id: str
    session_id: str
    
    # Content types
    content: Union[TextContent, AudioContent, EventContent]
    
    # Metadata
    metadata: MsgMetadata
    
    # Direction
    direction: MsgDirection  # INBOUND | OUTBOUND


@dataclass
class TextContent:
    text: str
    role: str = "user"  # "user" | "assistant"


@dataclass  
class AudioContent:
    audio_data: bytes  # raw audio bytes
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class EventContent:
    event_type: str
    payload: dict


@dataclass
class MsgMetadata:
    """Metadata about the message."""
    transport: str  # "websocket", "webrtc", "rest"
    client_info: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MsgDirection(str, Enum):
    INBOUND = "inbound"   # Client → Core
    OUTBOUND = "outbound" # Core → Client
```

### CoreInterface Methods

```python
class CoreInterface(ABC):
    """Interface that the transport layer uses to communicate with the core."""
    
    @abstractmethod
    async def process_message(self, msg: CoreMsg) -> AsyncGenerator[CoreMsg, None]:
        """Send a message to the core and get responses."""
        ...
    
    @abstractmethod
    async def send_notification(self, user_id: str, notification: CoreMsg) -> None:
        """Send a notification to a specific user."""
        ...
    
    @abstractmethod
    async def health_check(self) -> dict:
        """Check core health."""
        ...
```

### CoreHandler (Implementation)

The CoreHandler wraps the existing processor pipeline:

```python
class CoreHandler:
    """Implements CoreInterface using existing Aether components."""
    
    def __init__(
        self,
        llm_provider,
        memory_store,
        tool_registry,
        skill_loader,
        plugin_context,
        stt_provider,
        tts_provider,
    ):
        ...
    
    async def process_message(self, msg: CoreMsg) -> AsyncGenerator[CoreMsg, None]:
        # Convert CoreMsg → Frame
        # Run through pipeline (Memory → LLM → TTS)
        # Yield CoreMsg responses
        ...
    
    async def send_notification(self, user_id: str, notification: CoreMsg) -> None:
        # Push to connected clients for this user
        ...
```

## Transport Interface

All transports implement a common interface:

```python
class Transport(ABC):
    """Base class for all transports."""
    
    name: str  # "websocket", "webrtc", "rest", etc.
    
    @abstractmethod
    async def start(self) -> None:
        """Start the transport (listen for connections)."""
        ...
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport."""
        ...
    
    @abstractmethod
    async def send(self, user_id: str, msg: CoreMsg) -> None:
        """Send a message to a specific user."""
        ...
    
    @abstractmethod
    async def broadcast(self, msg: CoreMsg) -> None:
        """Broadcast a message to all connected users."""
        ...
    
    @abstractmethod
    async def get_connected_users(self) -> list[str]:
        """Get list of connected user IDs."""
        ...
    
    # Transport registers a callback to receive messages from clients
    def on_message(self, callback: Callable[[CoreMsg], Awaitable[None]]) -> None:
        """Set callback for incoming messages."""
        ...
    
    # Transport notifies manager of connection changes
    def on_connection_change(
        self, 
        callback: Callable[[str, ConnectionState], Awaitable[None]]
    ) -> None:
        """Set callback for connection state changes."""
        ...


class ConnectionState(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTED = "reconnected"
```

## TransportManager

The TransportManager is the facade that orchestrates everything:

```python
class TransportManager:
    """Central manager for all transports."""
    
    def __init__(self, core: CoreInterface):
        self.core = core
        self.transports: dict[str, Transport] = {}
        self.user_connections: dict[str, set[str]] = {}  # user_id -> session_ids
    
    async def register_transport(self, transport: Transport) -> None:
        """Register a transport."""
        ...
    
    async def start_all(self) -> None:
        """Start all registered transports."""
        ...
    
    async def stop_all(self) -> None:
        """Stop all transports."""
        ...
    
    async def send_to_user(self, user_id: str, msg: CoreMsg) -> None:
        """Send message to specific user (across all their connections)."""
        ...
    
    async def broadcast(self, msg: CoreMsg) -> None:
        """Broadcast message to all connected users."""
        ...
    
    async def handle_incoming(self, msg: CoreMsg) -> AsyncGenerator[CoreMsg, None]:
        """Process incoming message from client through core."""
        ...
    
    async def get_status(self) -> dict:
        """Get status of all transports."""
        ...
```

## Client Protocol (Current)

The current protocol from main.py:

### Client → Server
| Message | Fields |
|---------|--------|
| `text` | `{"type": "text", "data": "hello"}` |
| `audio` | `{"type": "audio", "data": "<base64>"}` |
| `audio_chunk` | `{"type": "audio_chunk", "data": "<base64>"}` |
| `stream_start` | `{"type": "stream_start"}` |
| `stream_stop` | `{"type": "stream_stop"}` |
| `mute` | `{"type": "mute"}` |
| `unmute` | `{"type": "unmute"}` |
| `image` | `{"type": "image", "data": "<base64>", "mime": "..."}` |
| `session_config` | `{"type": "session_config", "mode": "voice\|text"}` |
| `notification_feedback` | `{"type": "notification_feedback", "data": {...}}` |

### Server → Client
| Message | Fields |
|---------|--------|
| `text_chunk` | `{"type": "text_chunk", "data": "sentence", "index": 0}` |
| `audio_chunk` | `{"type": "audio_chunk", "data": "<base64>", "index": 0}` |
| `transcript` | `{"type": "transcript", "data": "text", "interim": true/false}` |
| `status` | `{"type": "status", "data": "..."}` |
| `notification` | `{"type": "notification", "data": {...}}` |
| `stream_end` | `{"type": "stream_end"}` |
| `tool_result` | `{"type": "tool_result", "data": {...}}` |

## File Structure

```
src/aether/transport/
├── __init__.py
├── core_msg.py        # CoreMsg, TextContent, AudioContent, etc.
├── interface.py       # CoreInterface ABC
├── handler.py         # CoreHandler implementation
├── manager.py         # TransportManager
├── base.py           # Transport ABC
├── websocket.py       # WebSocketTransport
├── webrtc.py         # WebRTCTransport (Daily/LiveKit)
├── push.py           # PushTransport (for notifications)
└── protocol.py        # Message serialization/deserialization
```

## Integration with main.py

The new main.py will look like:

```python
# main.py

# Initialize core components (unchanged)
memory_store = MemoryStore()
stt_provider = get_stt_provider()
llm_provider = get_llm_provider()
tts_provider = get_tts_provider()
tool_registry = ToolRegistry(...)
skill_loader = SkillLoader(...)
plugin_context = PluginContextStore(...)

# Create core interface
core = CoreHandler(
    llm_provider=llm_provider,
    memory_store=memory_store,
    tool_registry=tool_registry,
    skill_loader=skill_loader,
    plugin_context=plugin_context,
    stt_provider=stt_provider,
    tts_provider=tts_provider,
)

# Create transport manager
manager = TransportManager(core=core)

# Register transports
manager.register_transport(WebSocketTransport(manager.handle_incoming))
manager.register_transport(WebRTCTransport(manager.handle_incoming))
manager.register_transport(PushTransport(manager.handle_incoming))

# Start all
await manager.start_all()

# Keep for REST endpoints (unchanged)
app = FastAPI(...)
```

## Backward Compatibility

The new transport layer must maintain backward compatibility:
1. WebSocket endpoint at `/ws` - same protocol
2. All existing REST endpoints work unchanged
3. Plugin events continue to work
4. Health checks remain the same

## Future Extensibility

Adding new transports is easy:

1. Create new transport class extending `Transport`
2. Register with `manager.register_transport()`
3. Implement the protocol translation

Example for Twilio:
```python
class TwilioTransport(Transport):
    name = "twilio"
    
    async def start(self):
        # Set up Twilio webhook endpoints
        ...
    
    async def send(self, user_id, msg):
        # Make Twilio API call
        ...
```
"""
