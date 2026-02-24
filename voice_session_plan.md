# Aether — Voice Session Plan: Gemini Live as P Worker

> This document defines the phased implementation plan for replacing the current `VoiceSession` pipeline (Deepgram STT → Claude/AgentCore → TTS) with a **Gemini Live Realtime Model** architecture. Gemini becomes the **P Worker** — the always-on, multi-modal front door for all user communication (voice, text, video). Complex tasks (~40% of requests) are delegated to **Claude Opus** (the E Worker) via the existing Task Ledger. Each phase is independently testable. Phases are ordered by dependency.

---

## Table of Contents

1. [The Gap](#1-the-gap)
2. [What Stays Unchanged](#2-what-stays-unchanged)
3. [What Gets Deleted](#3-what-gets-deleted)
4. [Phase Overview](#4-phase-overview)
5. [Phase 1 — AudioIO Protocol + RealtimeModel ABCs](#5-phase-1--audioio-protocol--realtimemodel-abcs)
6. [Phase 2 — VoiceSession Rewrite](#6-phase-2--voicesession-rewrite)
7. [Phase 3 — Transport Update (WebRTC + Telephony)](#7-phase-3--transport-update-webrtc--telephony)
8. [Phase 4 — Gemini Realtime Backend](#8-phase-4--gemini-realtime-backend)
9. [Phase 5 — P-Tools Registry](#9-phase-5--p-tools-registry)
10. [Phase 6 — Delegation Bridge (P→E)](#10-phase-6--delegation-bridge-pe)
11. [Phase 7 — Factory + Wiring](#11-phase-7--factory--wiring)
12. [Summary](#12-summary)

---

## 1. The Gap

The current voice pipeline is a classic STT → LLM → TTS chain. Every user utterance is transcribed by Deepgram, sent to Claude via AgentCore, and the response is synthesized back to audio. This architecture has three fundamental problems:

### Current Model

```
Mic audio → VAD → Deepgram STT → transcript → EOU state machine → turn detector
    → AgentCore.generate_reply_voice() (Claude) → TTS → speaker audio

Components involved:
  - VoiceSession (1109 lines): STT lifecycle, EOU state machine, VAD-gated
    STT connect/disconnect, barge-in, echo suppression, dialog history,
    notification queue, watchdog recovery
  - DeepgramSTTProvider: WebSocket connection to Deepgram
  - TurnDetector: LiveKit ONNX model for endpointing
  - AgentCore: Claude via OpenRouter (every utterance, every time)
  - TTSProvider: OpenAI TTS for synthesis
```

### Problems

**1. Every utterance goes to Claude.** "What time is it?" and "Draft a 3-email sequence for the client" both take the same path through Claude. Simple requests (~60% of traffic) don't need a reasoning model — they need fast, direct answers.

**2. Four serial hops for voice.** Mic → Deepgram STT → Claude → OpenAI TTS → Speaker. Each hop adds latency. Gemini Live handles speech-in → speech-out natively in a single model call with ~300ms latency.

**3. No multi-modal path.** The current pipeline is audio-only. Adding video, screen sharing, or image understanding would require bolting on more services. Gemini Live handles audio, video, and text natively.

**4. Tight coupling.** VoiceSession is 1109 lines of STT lifecycle management, EOU state machines, VAD-gated connect/disconnect, and turn detection — all tightly coupled to Deepgram and the specific STT→LLM→TTS flow. Swapping to a different realtime model requires rewriting the entire file.

### Target Model

**All user communication flows through Gemini (P Worker).** Users never interact with Claude directly. Claude is the E Worker — it receives structured tasks and returns structured results. Gemini is the single front door for voice, text, and video.

```
Voice:   Mic audio ──→ AudioIO ──→ VoiceSession ──→ RealtimeSession (Gemini Live)
Text:    REST/WS ──→ RealtimeSession.generate_reply(text) ──→ Gemini Live
Video:   Camera ──→ RealtimeSession (Gemini Live, native multimodal)
                                               │
                                               ├── Simple request (60%): Gemini answers directly
                                                │   via ToolBridge (any tool — time, weather, memory, email...)
                                               │
                                               └── Complex request (40%): delegate_to_agent()
                                                   → Task Ledger → E Worker (Claude Opus)
                                                   → result fed back to Gemini → spoken/written to user
                                               │
Voice:   Speaker audio ←── AudioIO ←── VoiceSession ←───┘
Text:    REST/WS response ←── RealtimeSession text events ←─┘

Three abstraction boundaries:
  1. io.py      — Transport ↔ VoiceSession (AudioIO protocol, voice path)
  2. realtime.py — VoiceSession ↔ Model backend (RealtimeModel/RealtimeSession ABC)
  3. REST/WS    — Text transports ↔ RealtimeSession.generate_reply() (text path)

Swapping Gemini for OpenAI Realtime (or back to classic STT→LLM→TTS):
  → Change AETHER_VOICE_BACKEND env var
  → Zero transport code changes
```

### The P Worker / E Worker Split

This plan implements the P Worker half of the architecture defined in `Requirements.md` §2.2. The E Worker (Phases 1–7 of `implementation_plan.md`) is already complete.

| Worker | Model | Role | Traffic |
|---|---|---|---|
| **P Worker** (this plan) | Gemini Live 2.5 Flash | Always-on presence. Voice/text/video front door. Handles simple tasks directly via ToolBridge. | ~60% handled directly |
| **E Worker** (done) | Claude Opus | Execution brain. Complex reasoning, multi-step tools, sub-agents, memory extraction. | ~40% delegated |

Communication: in-process via Task Ledger. Same container, single-user architecture.

---

## 2. What Stays Unchanged

These components are not modified in any phase. They are the stable foundation.

| Component | Location | Why Unchanged |
|---|---|---|
| `AgentCore` | `src/aether/agent.py` | E Worker entry point — called via bridge, not directly from users |
| `LLMCore` / `ContextBuilder` | `src/aether/llm/` | E Worker's inner loop — unchanged |
| `SessionStore` / `TaskLedger` | `src/aether/session/` | P↔E communication channel — used by bridge |
| `SessionLoop` / `SubAgentManager` | `src/aether/session/`, `src/aether/agents/` | E Worker's outer loop — unchanged |
| `EventStream` | `src/aether/kernel/event_bus.py` | Event streaming — unchanged |
| `ToolRegistry` / `ToolOrchestrator` | `src/aether/tools/` | E Worker tool execution — unchanged |
| All existing tools | `src/aether/tools/` | E Worker tools — unchanged (ToolBridge exposes them to Gemini) |
| All existing plugins | `src/aether/plugins/` | Plugin system — unchanged |
| `TelephonyProtocol` / codec helpers | `src/aether/voice/telephony_protocol.py` | Pure transport concern — unchanged |
| `SileroVAD` | `src/aether/voice/vad.py` | Still useful for transport-level pre-speech detection |
| Session HTTP endpoints | `src/aether/http/sessions.py` | Session management API — unchanged (used by dashboard) |
| Memory system | `src/aether/memory/` | Three-bucket model — unchanged |
| Skill system | `src/aether/skills/` | Skill loader + tools — unchanged |
| Config system (existing configs) | `src/aether/core/config.py` | Existing configs stay, new one added |

**What changes for text:** `POST /v1/chat/completions` and `POST /chat` no longer go directly to Claude. They route through the P Worker (Gemini) via `RealtimeSession.generate_reply(text)`. Gemini handles the response — either directly via ToolBridge or by delegating to Claude. The user always talks to Gemini, regardless of modality.

---

## 3. What Gets Deleted

These components are fully replaced by Gemini Live's native capabilities.

| Component | Location | Lines | Why Deleted |
|---|---|---|---|
| STT lifecycle management | `voice/session.py` | ~200 | Gemini handles speech recognition natively |
| EOU state machine | `voice/session.py` | ~150 | Gemini handles end-of-utterance natively |
| VAD-gated STT connect/disconnect | `voice/session.py` | ~100 | Gemini receives continuous audio, handles VAD internally |
| Turn detection logic | `voice/session.py` | ~80 | Gemini handles turn detection natively |
| TTS synthesis pipeline | `voice/session.py` | ~120 | Gemini outputs audio directly |
| Barge-in handling | `voice/session.py` | ~60 | Gemini handles interruption natively |
| Echo suppression | `voice/session.py` | ~40 | Not needed — Gemini manages its own output |
| Watchdog recovery | `voice/session.py` | ~60 | Replaced by simpler health check on WebSocket |
| `DeepgramSTTProvider` usage | `voice/session.py` | ~80 | No more STT provider |
| `TurnDetector` | `voice/turn_detection.py` | 208 | **Entire file deleted** — Gemini handles turn detection |
| `TurnDetectionConfig` | `core/config.py` | ~35 | Config for deleted turn detector |
| `STTConfig` | `core/config.py` | ~30 | Config for deleted STT provider (kept but unused by voice) |

**Total deleted from VoiceSession**: ~890 of 1109 lines. The remaining ~220 lines of session lifecycle (create, destroy, pause, resume, session_id management) are preserved and simplified.

---

## 4. Phase Overview

| Phase | Name | Depends On | New Files | Modified Files | New Lines | Modified Lines | Status |
|---|---|---|---|---|---|---|---|
| 1 | AudioIO Protocol + RealtimeModel ABCs | — | 2 | 0 | ~200 | 0 | Pending |
| 2 | VoiceSession Rewrite | Phase 1 | 0 | 1 | 0 | ~900 (rewrite) | Pending |
| 3 | Transport Update | Phases 1, 2 | 0 | 2 | ~30 | ~120 | Pending |
| 4 | Gemini Realtime Backend | Phase 1 | 3 | 0 | ~600 | 0 | Pending |
| 5 | P-Tools Registry | Phase 4 | 1 | 0 | ~250 | 0 | Pending |
| 6 | Delegation Bridge (P→E) | Phases 4, 5 | 1 | 0 | ~200 | 0 | Pending |
| 7 | Factory + Wiring | Phases 1–6 | 1 | 2 | ~80 | ~60 | Pending |
| | **Total** | | **8** | **5** | **~1,360** | **~1,080** | |

**Net effect**: VoiceSession goes from 1109 lines to ~200 lines. Turn detection file (208 lines) is deleted. ~1,360 new lines across 8 new files. The voice pipeline becomes a thin orchestrator over a pluggable realtime model.

### New Directory Structure

```
app/src/aether/voice/
├── io.py                    ← NEW (Phase 1): AudioIO protocol
├── realtime.py              ← NEW (Phase 1): RealtimeModel + RealtimeSession ABC
├── session.py               ← REWRITTEN (Phase 2): thin orchestrator (~200 lines)
├── webrtc.py                ← MODIFIED (Phase 3): use AudioIO protocol
├── telephony.py             ← MODIFIED (Phase 3): use AudioIO protocol
├── backends/
│   └── gemini/
│       ├── __init__.py      ← NEW (Phase 4): exports
│       ├── model.py         ← NEW (Phase 4): GeminiRealtimeModel
│       ├── session.py       ← NEW (Phase 4): GeminiRealtimeSession
│       ├── tool_bridge.py   ← NEW (Phase 5): ToolBridge — all tools for Gemini + ledger logging
│       └── bridge.py        ← NEW (Phase 6): delegate_to_agent() → E Worker
├── factory.py               ← NEW (Phase 7): creates RealtimeModel from config
├── vad.py                   ← KEPT: transport pre-speech detection
├── telephony_protocol.py    ← KEPT: codec/encoding helpers
└── turn_detection.py        ← DELETED (Phase 2): Gemini handles natively
```

---

## 5. Phase 1 — AudioIO Protocol + RealtimeModel ABCs

**Why first**: These are the two abstraction boundaries that everything else depends on. No implementation code — just protocols and ABCs. Once defined, VoiceSession, transports, and backends can all be developed against stable interfaces.

### New Files

```
src/aether/voice/io.py         — AudioIO protocol (transport ↔ VoiceSession boundary)
src/aether/voice/realtime.py   — RealtimeModel + RealtimeSession ABC (VoiceSession ↔ backend)
```

### `io.py` — AudioIO Protocol

The AudioIO protocol defines how transports (WebRTC, telephony, future REST) communicate with VoiceSession. It replaces the current ad-hoc callback wiring where transports reach into VoiceSession internals (`on_audio_in`, `on_audio_out`, `on_barge_in`, `on_vad_event`, `on_text_event`).

```python
"""
AudioIO — transport-agnostic I/O protocol.

Transports implement AudioInput and AudioOutput.
VoiceSession consumes AudioInput and produces to AudioOutput.
Neither side knows about the other's implementation.

Audio format contract:
  - Input:  16kHz, 16-bit signed PCM, mono, little-endian (bytes)
  - Output: 24kHz, 16-bit signed PCM, mono, little-endian (bytes)
  
Transports are responsible for resampling to/from their native format
(e.g., WebRTC 48kHz, telephony 8kHz mulaw) before passing through AudioIO.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class AudioInputEvent(Enum):
    """Events that AudioInput can emit alongside audio data."""
    START_OF_SPEECH = "start_of_speech"
    END_OF_SPEECH = "end_of_speech"


@dataclass
class AudioFrame:
    """A chunk of PCM audio with optional metadata."""
    data: bytes                          # Raw PCM bytes
    sample_rate: int                     # Hz (16000 for input, 24000 for output)
    samples_per_channel: int             # Number of samples in this frame
    event: AudioInputEvent | None = None # Optional speech boundary event


class AudioInput(ABC):
    """
    Audio source — implemented by transports.
    
    Yields AudioFrames from the user's microphone (or phone line).
    Frames arrive at the transport's native rate, already resampled to 16kHz.
    """

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[AudioFrame]:
        """Async iterator yielding audio frames from the user."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Stop producing frames. Called on session teardown."""
        ...


class AudioOutput(ABC):
    """
    Audio sink — implemented by transports.
    
    Receives AudioFrames to play to the user's speaker (or phone line).
    Frames arrive at 24kHz from the model, transport resamples to native rate.
    """

    @abstractmethod
    async def push_frame(self, frame: AudioFrame) -> None:
        """Send an audio frame to the user. Non-blocking."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear any buffered audio (barge-in). Non-blocking."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Stop accepting frames. Called on session teardown."""
        ...


class TextOutput(ABC):
    """
    Text sink — implemented by transports that support text display.
    
    WebRTC data channel, telephony (no-op), future REST transport.
    Used for real-time transcript display, status messages, etc.
    """

    @abstractmethod
    async def push_text(self, text: str, *, final: bool = False) -> None:
        """Send text to the client. `final=True` marks end of a message."""
        ...

    @abstractmethod
    async def push_state(self, state: str) -> None:
        """Send agent state change (e.g., 'thinking', 'speaking')."""
        ...


class NullTextOutput(TextOutput):
    """No-op text output for transports that don't support text (telephony)."""

    async def push_text(self, text: str, *, final: bool = False) -> None:
        pass

    async def push_state(self, state: str) -> None:
        pass
```

**Design decisions**:
- `AudioInput` is an async iterator, not a callback. This lets VoiceSession pull frames at its own pace and naturally handles backpressure.
- `AudioOutput.clear()` replaces the current barge-in callback. When the model detects an interruption, VoiceSession calls `clear()` on the output to flush buffered audio.
- `TextOutput` is separate from `AudioOutput` because not all transports support text (telephony doesn't). `NullTextOutput` is the default.
- Audio format is fixed at the boundary: 16kHz in, 24kHz out. Transports handle their own resampling. This matches Gemini Live's native rates (`INPUT_AUDIO_SAMPLE_RATE = 16000`, `OUTPUT_AUDIO_SAMPLE_RATE = 24000` from the LiveKit plugin).

### `realtime.py` — RealtimeModel + RealtimeSession ABC

The RealtimeModel ABC defines how VoiceSession creates and manages realtime model sessions. This is the backend-agnostic interface that Gemini, OpenAI Realtime, or any future model implements.

```python
"""
RealtimeModel / RealtimeSession — backend-agnostic realtime model interface.

Follows the LiveKit pattern (livekit/agents/llm/realtime.py) adapted for Aether.

RealtimeModel is a factory — it creates RealtimeSessions.
RealtimeSession is the live connection — it receives audio, emits audio/text,
handles function calls, and manages the conversation.

VoiceSession owns one RealtimeSession at a time. The session is created
when a user connects and destroyed when they disconnect.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Awaitable


# ─── Events emitted by RealtimeSession ───────────────────────

class RealtimeEventType(Enum):
    """Events that a RealtimeSession can emit."""
    # Audio
    AUDIO_DELTA = "audio_delta"           # Model is producing audio
    AUDIO_DONE = "audio_done"             # Model finished producing audio for this turn
    
    # Text
    TEXT_DELTA = "text_delta"             # Model is producing text (transcript of its speech)
    TEXT_DONE = "text_done"               # Model finished producing text for this turn
    
    # Conversation
    INPUT_SPEECH_STARTED = "input_speech_started"     # User started speaking
    INPUT_SPEECH_STOPPED = "input_speech_stopped"     # User stopped speaking
    INPUT_SPEECH_COMMITTED = "input_speech_committed" # User's speech was committed as a turn
    INPUT_SPEECH_TRANSCRIPTION_COMPLETED = "input_speech_transcription_completed"
    
    # Function calls
    FUNCTION_CALL = "function_call"       # Model wants to call a function
    FUNCTION_CALL_DONE = "function_call_done"  # Function call completed (all args received)
    
    # Session
    SESSION_CREATED = "session_created"
    SESSION_RESUMED = "session_resumed"   # Session resumed after disconnect
    SESSION_ERROR = "session_error"
    
    # Turn
    TURN_STARTED = "turn_started"         # Model started a new turn
    TURN_DONE = "turn_done"               # Model finished its turn (audio + text + function calls)
    
    # Interruption
    INTERRUPTED = "interrupted"           # User interrupted the model


@dataclass
class RealtimeEvent:
    """An event emitted by a RealtimeSession."""
    type: RealtimeEventType
    data: dict[str, Any] = field(default_factory=dict)
    # For AUDIO_DELTA: {"audio": bytes, "sample_rate": int}
    # For TEXT_DELTA: {"text": str}
    # For FUNCTION_CALL: {"call_id": str, "name": str, "arguments": str}
    # For FUNCTION_CALL_DONE: {"call_id": str, "name": str, "arguments": str}
    # For INPUT_SPEECH_TRANSCRIPTION_COMPLETED: {"transcript": str}
    # For SESSION_ERROR: {"error": str}


@dataclass
class FunctionCallResult:
    """Result of a function call to feed back to the model."""
    call_id: str
    name: str
    result: str  # JSON-encoded result string


@dataclass
class RealtimeModelConfig:
    """Configuration for creating a RealtimeModel."""
    model: str = ""                       # e.g., "gemini-2.5-flash-preview-native-audio"
    api_key: str = ""
    voice: str = "Puck"                   # Gemini voice name
    instructions: str = ""                # System instructions
    temperature: float = 0.7
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    # Gemini-specific
    language: str = "en"
    response_modalities: list[str] = field(default_factory=lambda: ["AUDIO"])
    # Session resumption
    enable_session_resumption: bool = True


# ─── ABCs ─────────────────────────────────────────────────────

class RealtimeSession(ABC):
    """
    A live connection to a realtime model.
    
    Lifecycle:
      1. Created by RealtimeModel.create_session()
      2. VoiceSession pushes audio via push_audio()
      3. Events are consumed via events() async iterator
      4. Function call results fed back via send_function_result()
      5. Destroyed via close()
    
    The session handles:
      - Continuous audio input (no VAD needed — model handles it)
      - Turn detection (model decides when user is done speaking)
      - Audio output generation (model speaks directly)
      - Function calling (model calls tools via ToolBridge, receives results)
      - Interruption handling (model detects barge-in)
      - Session resumption (reconnect without losing context)
    """

    @abstractmethod
    async def push_audio(self, data: bytes) -> None:
        """Push raw PCM audio from the user to the model.
        
        Audio format: 16kHz, 16-bit signed PCM, mono, little-endian.
        Called continuously while the user's mic is active.
        The model handles VAD and turn detection internally.
        """
        ...

    @abstractmethod
    def events(self) -> AsyncIterator[RealtimeEvent]:
        """Async iterator of events from the model.
        
        VoiceSession consumes this to route audio to AudioOutput,
        text to TextOutput, and function calls to the ToolBridge.
        
        The iterator ends when the session is closed.
        """
        ...

    @abstractmethod
    async def send_function_result(self, result: FunctionCallResult) -> None:
        """Feed a function call result back to the model.
        
        Called after a tool executes. The model incorporates the result
        and continues generating its response (possibly calling more tools).
        """
        ...

    @abstractmethod
    async def interrupt(self) -> None:
        """Signal that the user interrupted the model.
        
        Called when AudioInput detects START_OF_SPEECH while the model
        is producing audio. The model should stop generating and listen.
        """
        ...

    @abstractmethod
    async def generate_reply(self, text: str) -> None:
        """Inject a text message and trigger a model response.
        
        Used for:
          - Text-only input (REST/WebSocket without audio)
          - Feeding E Worker results back as text for the model to speak
          - System notifications ("You have a new email")
        """
        ...

    @abstractmethod
    async def update_instructions(self, instructions: str) -> None:
        """Update the system instructions mid-session.
        
        Used when user preferences change or context needs updating.
        """
        ...

    @abstractmethod
    async def update_tools(self, tools: list[dict[str, Any]]) -> None:
        """Update the available function declarations mid-session.
        
        Used when tools change (e.g., plugin loaded/unloaded).
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the session and release resources.
        
        If session resumption is enabled, the session token is preserved
        so a new session can resume the conversation.
        """
        ...

    @property
    @abstractmethod
    def session_id(self) -> str:
        """Unique identifier for this realtime session."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the WebSocket to the model is alive."""
        ...

    @property
    @abstractmethod
    def resumption_token(self) -> str | None:
        """Token for resuming this session after disconnect. None if not supported."""
        ...


class RealtimeModel(ABC):
    """
    Factory for RealtimeSession instances.
    
    One RealtimeModel per application. Creates sessions on demand.
    Holds shared config (API key, model name, default instructions).
    """

    @abstractmethod
    async def create_session(
        self,
        *,
        instructions: str = "",
        tools: list[dict[str, Any]] | None = None,
        resumption_token: str | None = None,
    ) -> RealtimeSession:
        """Create a new realtime session.
        
        Args:
            instructions: System instructions for this session.
            tools: Function declarations (Gemini tool format).
            resumption_token: If provided, resume a previous session.
        
        Returns:
            A connected RealtimeSession ready to receive audio.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Shut down the model and release shared resources."""
        ...

    @property
    @abstractmethod
    def config(self) -> RealtimeModelConfig:
        """The model configuration."""
        ...
```

**Design decisions**:
- **Event-driven, not callback-driven.** `RealtimeSession.events()` returns an async iterator. VoiceSession consumes it in a single event loop. This is cleaner than registering N callbacks and matches the LiveKit pattern.
- **`push_audio()` is fire-and-forget.** Audio is pushed continuously. The model handles VAD, turn detection, and endpointing internally. No more EOU state machine.
- **`generate_reply(text)` for text injection.** This is how E Worker results get spoken — the bridge calls `generate_reply("Here's what I found: ...")` and Gemini speaks it.
- **`update_tools()` and `update_instructions()` for mid-session changes.** Gemini Live supports updating tools and instructions without reconnecting.
- **`resumption_token` for session continuity.** Gemini Live supports session resumption — if the WebSocket drops, a new session can pick up where the old one left off without losing conversation context.
- **No `commit_audio()` / `clear_audio()`.** Unlike LiveKit's ABC which supports both realtime and classic pipelines, we only support realtime models. The model decides when to commit audio (turn detection is internal).

### Acceptance Criteria

- [ ] `AudioInput`, `AudioOutput`, `TextOutput` protocols are importable and type-checkable
- [ ] `RealtimeModel`, `RealtimeSession` ABCs are importable and type-checkable
- [ ] All event types are defined in `RealtimeEventType`
- [ ] `NullTextOutput` passes type checking as a `TextOutput`
- [ ] No runtime dependencies — these files import only stdlib + typing

### Estimated Size

~200 new lines across 2 files, 0 modified lines

---

## 6. Phase 2 — VoiceSession Rewrite

**Why second**: With the ABCs defined, VoiceSession can be rewritten as a thin orchestrator. This is the largest single change — the 1109-line monolith becomes ~200 lines.

### Modified Files

```
src/aether/voice/session.py    — Full rewrite (1109 → ~200 lines)
```

### Deleted Files

```
src/aether/voice/turn_detection.py   — 208 lines, entirely deleted
```

### What VoiceSession Becomes

The new VoiceSession is a thin orchestrator with three responsibilities:

1. **Audio routing**: Read from `AudioInput`, push to `RealtimeSession.push_audio()`. Read audio events from `RealtimeSession.events()`, push to `AudioOutput`.
2. **Event dispatch**: Consume `RealtimeEvent`s and route them — audio to `AudioOutput`, text to `TextOutput`, function calls to the ToolBridge.
3. **Lifecycle management**: Create/destroy `RealtimeSession`, handle pause/resume, manage session ID.

Everything else — STT, turn detection, EOU state machine, VAD-gated connect/disconnect, TTS synthesis, barge-in logic, echo suppression, watchdog — is **deleted**. Gemini handles all of it natively.

### Interface

```python
"""
VoiceSession — thin orchestrator over a RealtimeSession.

Routes audio between transport (AudioIO) and model (RealtimeSession).
Dispatches model events (audio, text, function calls) to the right outputs.

~200 lines. Was 1109.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from aether.voice.io import AudioFrame, AudioInput, AudioOutput, TextOutput, NullTextOutput
from aether.voice.realtime import (
    RealtimeEvent,
    RealtimeEventType,
    RealtimeModel,
    RealtimeSession,
    FunctionCallResult,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VoiceSession:
    """
    Owns the voice pipeline for one user connection.
    
    One instance per user (persistent across reconnects).
    Created by the transport, wired to AudioIO and RealtimeModel.
    """

    def __init__(
        self,
        session_id: str,
        realtime_model: RealtimeModel,
        *,
        on_function_call: Callable[[str, str, str], Awaitable[str]] | None = None,
        instructions: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.session_id = session_id
        self._model = realtime_model
        self._session: RealtimeSession | None = None
        self._on_function_call = on_function_call
        self._instructions = instructions
        self._tools = tools

        # I/O — set by transport before start()
        self._audio_input: AudioInput | None = None
        self._audio_output: AudioOutput | None = None
        self._text_output: TextOutput = NullTextOutput()

        # Tasks
        self._input_task: asyncio.Task | None = None
        self._event_task: asyncio.Task | None = None
        self._running = False

        # Session resumption
        self._resumption_token: str | None = None

    def set_io(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        text_output: TextOutput | None = None,
    ) -> None:
        """Set I/O channels. Called by transport before start()."""
        self._audio_input = audio_input
        self._audio_output = audio_output
        self._text_output = text_output or NullTextOutput()

    async def start(self) -> None:
        """Create RealtimeSession and start audio routing loops."""
        if self._running:
            return
        assert self._audio_input is not None, "set_io() must be called before start()"
        assert self._audio_output is not None, "set_io() must be called before start()"

        self._session = await self._model.create_session(
            instructions=self._instructions,
            tools=self._tools,
            resumption_token=self._resumption_token,
        )
        self._running = True
        self._input_task = asyncio.create_task(self._audio_input_loop())
        self._event_task = asyncio.create_task(self._event_dispatch_loop())
        logger.info("VoiceSession %s started (model session %s)", self.session_id, self._session.session_id)

    async def stop(self) -> None:
        """Stop routing and close the RealtimeSession."""
        self._running = False
        if self._session:
            self._resumption_token = self._session.resumption_token
            await self._session.close()
            self._session = None
        for task in (self._input_task, self._event_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._input_task = None
        self._event_task = None
        logger.info("VoiceSession %s stopped", self.session_id)

    async def pause(self) -> None:
        """Pause on disconnect — preserve resumption token."""
        await self.stop()

    async def resume(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        text_output: TextOutput | None = None,
    ) -> None:
        """Resume on reconnect — reuse resumption token."""
        self.set_io(audio_input, audio_output, text_output)
        await self.start()

    async def inject_text(self, text: str) -> None:
        """Inject text for the model to respond to (notifications, E Worker results)."""
        if self._session:
            await self._session.generate_reply(text)

    # ─── Internal loops ───────────────────────────────────────

    async def _audio_input_loop(self) -> None:
        """Read audio from transport, push to model."""
        try:
            async for frame in self._audio_input:
                if not self._running:
                    break
                await self._session.push_audio(frame.data)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Audio input loop error in session %s", self.session_id)

    async def _event_dispatch_loop(self) -> None:
        """Consume model events, route to outputs."""
        try:
            async for event in self._session.events():
                if not self._running:
                    break
                await self._handle_event(event)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Event dispatch loop error in session %s", self.session_id)

    async def _handle_event(self, event: RealtimeEvent) -> None:
        """Route a single event to the appropriate output."""
        match event.type:
            # Audio output
            case RealtimeEventType.AUDIO_DELTA:
                frame = AudioFrame(
                    data=event.data["audio"],
                    sample_rate=event.data.get("sample_rate", 24000),
                    samples_per_channel=len(event.data["audio"]) // 2,
                )
                await self._audio_output.push_frame(frame)

            case RealtimeEventType.AUDIO_DONE:
                pass  # No action needed — audio stream naturally ends

            # Text output (transcript of model's speech)
            case RealtimeEventType.TEXT_DELTA:
                await self._text_output.push_text(event.data.get("text", ""))

            case RealtimeEventType.TEXT_DONE:
                await self._text_output.push_text("", final=True)

            # User speech events
            case RealtimeEventType.INPUT_SPEECH_STARTED:
                await self._audio_output.clear()  # Barge-in: clear buffered audio
                await self._text_output.push_state("listening")

            case RealtimeEventType.INPUT_SPEECH_COMMITTED:
                await self._text_output.push_state("thinking")

            case RealtimeEventType.INPUT_SPEECH_TRANSCRIPTION_COMPLETED:
                transcript = event.data.get("transcript", "")
                if transcript:
                    await self._text_output.push_text(f"[user]: {transcript}", final=True)

            # Function calls
            case RealtimeEventType.FUNCTION_CALL_DONE:
                call_id = event.data["call_id"]
                name = event.data["name"]
                arguments = event.data["arguments"]
                await self._handle_function_call(call_id, name, arguments)

            # Turn lifecycle
            case RealtimeEventType.TURN_STARTED:
                await self._text_output.push_state("speaking")

            case RealtimeEventType.TURN_DONE:
                await self._text_output.push_state("idle")

            # Interruption
            case RealtimeEventType.INTERRUPTED:
                await self._audio_output.clear()

            # Errors
            case RealtimeEventType.SESSION_ERROR:
                logger.error("Realtime session error: %s", event.data.get("error", "unknown"))

    async def _handle_function_call(self, call_id: str, name: str, arguments: str) -> None:
        """Execute a function call and feed the result back to the model."""
        if not self._on_function_call:
            logger.warning("Function call %s(%s) but no handler registered", name, call_id)
            result = '{"error": "no function handler registered"}'
        else:
            try:
                result = await self._on_function_call(call_id, name, arguments)
            except Exception as e:
                logger.exception("Function call %s failed", name)
                result = f'{{"error": "{e!s}"}}'

        if self._session:
            await self._session.send_function_result(
                FunctionCallResult(call_id=call_id, name=name, result=result)
            )
```

### What Was Deleted (by category)

| Category | Old Code | Replacement |
|---|---|---|
| STT lifecycle (`_connect_stt`, `_disconnect_stt`, `_on_stt_*`) | ~200 lines | Gemini receives raw audio directly |
| EOU state machine (`_run_eou`, `_eou_task`, `_eou_token`) | ~150 lines | Gemini handles turn detection internally |
| VAD-gated STT (`_on_vad_start`, `_on_vad_end`, state transitions) | ~100 lines | Gemini handles VAD internally |
| Turn detection (`_run_turn_detector`, delay calculation) | ~80 lines | Gemini handles endpointing internally |
| TTS synthesis (`_synthesize_and_play`, `_play_audio_chunk`) | ~120 lines | Gemini outputs audio directly |
| Barge-in handling (`_handle_barge_in`, `_cancel_tts`) | ~60 lines | `INTERRUPTED` event → `AudioOutput.clear()` |
| Echo suppression (`_is_echo`, `_echo_buffer`) | ~40 lines | Not needed — model manages its own output |
| Watchdog (`_watchdog_loop`, stuck state recovery) | ~60 lines | Simple `is_connected` check on RealtimeSession |
| Dialog history management (`_dialog_history`, `_add_to_history`) | ~50 lines | Gemini maintains conversation context internally |
| Notification queue (`_notification_queue`, `_process_notifications`) | ~30 lines | `inject_text()` → `generate_reply()` |

### Acceptance Criteria

- [ ] VoiceSession is ≤250 lines
- [ ] `turn_detection.py` is deleted
- [ ] No imports of `DeepgramSTTProvider`, `TurnDetector`, or `TTSProvider` in `session.py`
- [ ] `VoiceSession.__init__` takes `RealtimeModel` instead of `AgentCore` + `TTSProvider`
- [ ] `set_io()` / `start()` / `stop()` / `pause()` / `resume()` lifecycle works
- [ ] `inject_text()` feeds text to the model for spoken delivery
- [ ] Function calls are dispatched to `on_function_call` handler
- [ ] Barge-in clears audio output on `INPUT_SPEECH_STARTED`
- [ ] All existing tests that reference VoiceSession internals are updated or removed

### Estimated Size

~200 lines (rewrite of 1109), 208 lines deleted (`turn_detection.py`)

---

## 7. Phase 3 — Transport Update (WebRTC + Telephony)

**Why third**: With AudioIO defined (Phase 1) and VoiceSession rewritten (Phase 2), transports need to implement the AudioIO protocol instead of reaching into VoiceSession internals.

### Modified Files

```
src/aether/voice/webrtc.py      — Implement AudioInput/AudioOutput, update VoiceSession wiring
src/aether/voice/telephony.py   — Implement AudioInput/AudioOutput, update VoiceSession wiring
```

### Changes to WebRTC Transport

The WebRTC transport currently:
1. Creates `VoiceSession(agent=agent_core, tts_provider=tts_provider, session_id=...)`
2. Sets callbacks directly: `vs.on_audio_out = ...`, `vs.on_barge_in = ...`, `vs.on_text_event = ...`
3. Calls `vs.on_audio_in(pcm_bytes)` when audio arrives
4. Calls `vs.on_vad_event("start"/"end")` when VAD fires

After this phase:
1. Creates `AudioInput` and `AudioOutput` implementations
2. Creates `VoiceSession(session_id=..., realtime_model=model)`
3. Calls `vs.set_io(audio_input, audio_output, text_output)`
4. Calls `vs.start()`

```python
# ─── WebRTC AudioInput ────────────────────────────────────────

class WebRTCAudioInput(AudioInput):
    """Wraps the aiortc audio track as an AudioInput."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=100)
        self._closed = False

    async def push(self, pcm_bytes: bytes, event: AudioInputEvent | None = None) -> None:
        """Called by the WebRTC audio track handler (internal)."""
        if self._closed:
            return
        frame = AudioFrame(
            data=pcm_bytes,
            sample_rate=16000,
            samples_per_channel=len(pcm_bytes) // 2,
            event=event,
        )
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # Drop frame if consumer is slow

    async def __aiter__(self):
        while not self._closed:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield frame
            except asyncio.TimeoutError:
                continue

    async def close(self) -> None:
        self._closed = True


# ─── WebRTC AudioOutput ───────────────────────────────────────

class WebRTCAudioOutput(AudioOutput):
    """Wraps the AudioOutputTrack as an AudioOutput."""

    def __init__(self, audio_out_track: "AudioOutputTrack") -> None:
        self._track = audio_out_track

    async def push_frame(self, frame: AudioFrame) -> None:
        """Resample 24kHz → 48kHz and enqueue for RTP pacing."""
        # Resampling logic already exists in AudioOutputTrack
        self._track.push_audio(frame.data, sample_rate=frame.sample_rate)

    async def clear(self) -> None:
        """Clear buffered audio (barge-in)."""
        self._track.clear()

    async def close(self) -> None:
        pass  # Track lifecycle managed by RTCPeerConnection


# ─── WebRTC TextOutput ────────────────────────────────────────

class WebRTCTextOutput(TextOutput):
    """Sends text/state via the WebRTC data channel."""

    def __init__(self, data_channel: Any) -> None:
        self._dc = data_channel

    async def push_text(self, text: str, *, final: bool = False) -> None:
        if self._dc and self._dc.readyState == "open":
            msg = json.dumps({"type": "text", "text": text, "final": final})
            self._dc.send(msg)

    async def push_state(self, state: str) -> None:
        if self._dc and self._dc.readyState == "open":
            msg = json.dumps({"type": "state", "state": state})
            self._dc.send(msg)
```

**Key changes in `WebRTCVoiceTransport`**:

```python
# BEFORE (current):
vs = VoiceSession(agent=self._agent, tts_provider=self._tts, session_id=session_id)
vs.on_audio_out = lambda audio: audio_out_track.push_audio(audio)
vs.on_text_event = lambda text, final: dc.send(json.dumps({...}))
# ... in audio handler:
await vs.on_audio_in(pcm_bytes)
await vs.on_vad_event("start")

# AFTER (new):
audio_input = WebRTCAudioInput()
audio_output = WebRTCAudioOutput(audio_out_track)
text_output = WebRTCTextOutput(data_channel)
vs = VoiceSession(session_id=session_id, realtime_model=self._realtime_model)
vs.set_io(audio_input, audio_output, text_output)
await vs.start()
# ... in audio handler:
await audio_input.push(pcm_bytes)
# VAD events from transport-level VAD (optional, for pre-speech detection):
await audio_input.push(b"", event=AudioInputEvent.START_OF_SPEECH)
```

### Changes to Telephony Transport

Same pattern. The telephony transport already resamples 8kHz mulaw → 16kHz PCM. It just needs to wrap that in `AudioInput`/`AudioOutput` instead of calling VoiceSession methods directly.

```python
class TelephonyAudioInput(AudioInput):
    """Wraps the telephony WebSocket audio as an AudioInput."""
    # Same pattern as WebRTCAudioInput — queue-based async iterator
    # Audio arrives as 8kHz mulaw, decoded and resampled to 16kHz before pushing

class TelephonyAudioOutput(AudioOutput):
    """Wraps the telephony WebSocket as an AudioOutput."""
    # push_frame: resample 24kHz → 8kHz, encode to mulaw, send via WebSocket
    # clear: send clear message via protocol adapter
```

**Telephony uses `NullTextOutput`** — phone calls don't have a text channel.

### What Stays the Same in Transports

- **WebRTC signaling** (`handle_offer`, `handle_ice_candidate`): unchanged
- **RTCPeerConnection lifecycle**: unchanged
- **AudioOutputTrack** (RTP pacing, 48kHz output): unchanged, just wrapped in `WebRTCAudioOutput`
- **Session persistence** (TTL, disconnect grace, reconnect): unchanged
- **Telephony protocol adapters** (Twilio/Telnyx/Vobiz): unchanged
- **Codec helpers** (`mulaw_to_pcm16`, `resample_*`): unchanged

### Acceptance Criteria

- [ ] WebRTC transport creates `AudioInput`/`AudioOutput`/`TextOutput` and passes to VoiceSession
- [ ] Telephony transport creates `AudioInput`/`AudioOutput` and passes to VoiceSession
- [ ] No direct calls to `VoiceSession.on_audio_in()`, `on_vad_event()`, `on_audio_out`, `on_text_event`
- [ ] WebRTC reconnect correctly calls `vs.resume()` with new I/O
- [ ] Telephony uses `NullTextOutput`
- [ ] Audio resampling still works correctly (48kHz↔16kHz for WebRTC, 8kHz↔16kHz for telephony)
- [ ] Transport-level VAD still works for pre-speech detection (optional, not required by model)

### Estimated Size

~30 new lines (AudioInput/AudioOutput implementations), ~120 modified lines (wiring changes)

---

## 8. Phase 4 — Gemini Realtime Backend

**Why fourth**: With ABCs defined (Phase 1), this phase implements the concrete Gemini Live backend. This is the largest new code — the actual WebSocket connection to Gemini, audio streaming, function call handling, and session resumption.

### New Files

```
src/aether/voice/backends/gemini/
├── __init__.py    — Exports GeminiRealtimeModel
├── model.py       — GeminiRealtimeModel implements RealtimeModel
└── session.py     — GeminiRealtimeSession implements RealtimeSession
```

### `model.py` — GeminiRealtimeModel

```python
"""
GeminiRealtimeModel — factory for Gemini Live sessions.

Uses the google-genai SDK to create live sessions with Gemini 2.5 Flash.
Holds shared config (API key, model name, default voice/instructions).
"""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types as genai_types

from aether.voice.realtime import RealtimeModel, RealtimeModelConfig, RealtimeSession

logger = logging.getLogger(__name__)


class GeminiRealtimeModel(RealtimeModel):
    """Creates Gemini Live realtime sessions."""

    def __init__(self, config: RealtimeModelConfig) -> None:
        self._config = config
        self._client = genai.Client(api_key=config.api_key)

    async def create_session(
        self,
        *,
        instructions: str = "",
        tools: list[dict[str, Any]] | None = None,
        resumption_token: str | None = None,
    ) -> RealtimeSession:
        """Create a new Gemini Live session."""
        from aether.voice.backends.gemini.session import GeminiRealtimeSession

        effective_instructions = instructions or self._config.instructions

        # Build Gemini Live config
        live_config = genai_types.LiveConnectConfig(
            response_modalities=self._config.response_modalities,
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=self._config.voice,
                    )
                )
            ),
            system_instruction=genai_types.Content(
                parts=[genai_types.Part(text=effective_instructions)]
            ),
            tools=self._build_tools(tools) if tools else None,
            session_resumption=genai_types.SessionResumptionConfig(
                handle=resumption_token,
            ) if self._config.enable_session_resumption else None,
            input_audio_transcription=genai_types.AudioTranscriptionConfig(),
            output_audio_transcription=genai_types.AudioTranscriptionConfig(),
        )

        # Connect to Gemini Live
        session = GeminiRealtimeSession(
            client=self._client,
            model=self._config.model,
            config=live_config,
        )
        await session.connect()
        return session

    async def close(self) -> None:
        """No shared resources to clean up."""
        pass

    @property
    def config(self) -> RealtimeModelConfig:
        return self._config

    def _build_tools(self, tools: list[dict[str, Any]]) -> list[genai_types.Tool]:
        """Convert tool declarations to Gemini format."""
        declarations = []
        for tool in tools:
            declarations.append(
                genai_types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters"),
                )
            )
        return [genai_types.Tool(function_declarations=declarations)]
```

### `session.py` — GeminiRealtimeSession

```python
"""
GeminiRealtimeSession — live connection to Gemini 2.5 Flash.

Manages the WebSocket connection, audio streaming, event parsing,
function call handling, and session resumption.

Audio format:
  - Input:  16kHz, 16-bit signed PCM, mono (from AudioInput via VoiceSession)
  - Output: 24kHz, 16-bit signed PCM, mono (to AudioOutput via VoiceSession)

Key implementation details (from LiveKit's Gemini plugin):
  - Audio is sent in chunks via send_realtime_input()
  - Responses arrive as server_content messages with inline_data (audio) or text
  - Function calls arrive as tool_call messages
  - Function results are sent back via send_tool_response()
  - Session resumption uses opaque tokens from the server
  - The server handles VAD, turn detection, and interruption natively
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from typing import Any, AsyncIterator

from google import genai
from google.genai import types as genai_types

from aether.voice.realtime import (
    FunctionCallResult,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSession,
)

logger = logging.getLogger(__name__)

INPUT_AUDIO_SAMPLE_RATE = 16000
OUTPUT_AUDIO_SAMPLE_RATE = 24000
AUDIO_CHUNK_DURATION_MS = 50  # 50ms chunks = 1600 bytes at 16kHz


class GeminiRealtimeSession(RealtimeSession):
    """Live WebSocket session with Gemini."""

    def __init__(
        self,
        client: genai.Client,
        model: str,
        config: genai_types.LiveConnectConfig,
    ) -> None:
        self._client = client
        self._model = model
        self._config = config
        self._session: Any = None  # google.genai.live.AsyncSession
        self._session_id = str(uuid.uuid4())
        self._connected = False
        self._resumption_token: str | None = None
        self._event_queue: asyncio.Queue[RealtimeEvent] = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self._closed = False

    async def connect(self) -> None:
        """Establish WebSocket connection to Gemini Live."""
        self._session = await self._client.aio.live.connect(
            model=self._model,
            config=self._config,
        ).__aenter__()
        self._connected = True
        self._recv_task = asyncio.create_task(self._receive_loop())
        logger.info("Gemini session %s connected", self._session_id)

    async def push_audio(self, data: bytes) -> None:
        """Send PCM audio to Gemini."""
        if not self._connected or not self._session:
            return
        try:
            await self._session.send_realtime_input(
                audio=genai_types.Blob(
                    data=data,
                    mime_type=f"audio/pcm;rate={INPUT_AUDIO_SAMPLE_RATE}",
                )
            )
        except Exception:
            logger.exception("Failed to send audio to Gemini")

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield events from the model."""
        while not self._closed:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                continue

    async def send_function_result(self, result: FunctionCallResult) -> None:
        """Send function call result back to Gemini."""
        if not self._connected or not self._session:
            return
        try:
            await self._session.send_tool_response(
                function_responses=[
                    genai_types.FunctionResponse(
                        name=result.name,
                        response={"result": result.result},
                        id=result.call_id,
                    )
                ]
            )
        except Exception:
            logger.exception("Failed to send function result to Gemini")

    async def interrupt(self) -> None:
        """Signal interruption to Gemini.
        
        Gemini handles interruption natively when it detects user speech
        during model output. This is a no-op — the model detects it from
        the audio stream. We emit INTERRUPTED locally for VoiceSession
        to clear the audio output buffer.
        """
        await self._event_queue.put(
            RealtimeEvent(type=RealtimeEventType.INTERRUPTED)
        )

    async def generate_reply(self, text: str) -> None:
        """Send text input and trigger a response."""
        if not self._connected or not self._session:
            return
        try:
            await self._session.send_client_content(
                turns=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=text)],
                ),
                turn_complete=True,
            )
        except Exception:
            logger.exception("Failed to send text to Gemini")

    async def update_instructions(self, instructions: str) -> None:
        """Update system instructions mid-session.
        
        Gemini Live supports this via session.update() or by reconnecting
        with new config. For now, we store and apply on next reconnect.
        """
        # Gemini Live API may support mid-session instruction updates
        # in future versions. For now, store for reconnection.
        self._config.system_instruction = genai_types.Content(
            parts=[genai_types.Part(text=instructions)]
        )
        logger.info("Instructions updated (will apply on next reconnect)")

    async def update_tools(self, tools: list[dict[str, Any]]) -> None:
        """Update available tools mid-session.
        
        Gemini Live supports tool updates via session configuration.
        """
        # Store for reconnection; live update support TBD based on API
        logger.info("Tools updated (%d tools)", len(tools))

    async def close(self) -> None:
        """Close the session."""
        self._closed = True
        self._connected = False
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error closing Gemini session")
            self._session = None
        logger.info("Gemini session %s closed", self._session_id)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def resumption_token(self) -> str | None:
        return self._resumption_token

    # ─── Internal receive loop ────────────────────────────────

    async def _receive_loop(self) -> None:
        """Read messages from Gemini and convert to RealtimeEvents."""
        try:
            async for message in self._session.receive():
                events = self._parse_message(message)
                for event in events:
                    await self._event_queue.put(event)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Gemini receive loop error")
            await self._event_queue.put(
                RealtimeEvent(
                    type=RealtimeEventType.SESSION_ERROR,
                    data={"error": "receive loop failed"},
                )
            )
        finally:
            self._connected = False

    def _parse_message(self, message: Any) -> list[RealtimeEvent]:
        """Parse a Gemini Live message into RealtimeEvents.
        
        Gemini Live messages have these key fields:
          - server_content.model_turn.parts[] — audio/text content
          - server_content.turn_complete — model finished its turn
          - server_content.interrupted — model was interrupted
          - tool_call — function call request
          - tool_call_cancellation — function call cancelled
          - session_resumption_update.new_handle — resumption token
          - setup_complete — session ready
          - go_away — server requesting disconnect
          - input_transcription — user speech transcript
        """
        events: list[RealtimeEvent] = []

        # Session resumption token update
        if hasattr(message, "session_resumption_update") and message.session_resumption_update:
            update = message.session_resumption_update
            if hasattr(update, "new_handle") and update.new_handle:
                self._resumption_token = update.new_handle

        # Setup complete
        if hasattr(message, "setup_complete") and message.setup_complete:
            events.append(RealtimeEvent(type=RealtimeEventType.SESSION_CREATED))
            return events

        # Server content (audio/text from model)
        if hasattr(message, "server_content") and message.server_content:
            sc = message.server_content

            # Check for interruption
            if hasattr(sc, "interrupted") and sc.interrupted:
                events.append(RealtimeEvent(type=RealtimeEventType.INTERRUPTED))
                return events

            # Model turn content
            if hasattr(sc, "model_turn") and sc.model_turn:
                for part in sc.model_turn.parts or []:
                    # Audio data
                    if hasattr(part, "inline_data") and part.inline_data:
                        audio_data = part.inline_data.data
                        if isinstance(audio_data, str):
                            audio_data = base64.b64decode(audio_data)
                        events.append(RealtimeEvent(
                            type=RealtimeEventType.AUDIO_DELTA,
                            data={
                                "audio": audio_data,
                                "sample_rate": OUTPUT_AUDIO_SAMPLE_RATE,
                            },
                        ))
                    # Text data
                    elif hasattr(part, "text") and part.text:
                        events.append(RealtimeEvent(
                            type=RealtimeEventType.TEXT_DELTA,
                            data={"text": part.text},
                        ))

            # Turn complete
            if hasattr(sc, "turn_complete") and sc.turn_complete:
                events.append(RealtimeEvent(type=RealtimeEventType.AUDIO_DONE))
                events.append(RealtimeEvent(type=RealtimeEventType.TEXT_DONE))
                events.append(RealtimeEvent(type=RealtimeEventType.TURN_DONE))

            # Input transcription (user's speech)
            if hasattr(sc, "input_transcription") and sc.input_transcription:
                transcript = sc.input_transcription.text if hasattr(sc.input_transcription, "text") else ""
                if transcript:
                    events.append(RealtimeEvent(
                        type=RealtimeEventType.INPUT_SPEECH_TRANSCRIPTION_COMPLETED,
                        data={"transcript": transcript},
                    ))

        # Tool calls
        if hasattr(message, "tool_call") and message.tool_call:
            for fc in message.tool_call.function_calls or []:
                events.append(RealtimeEvent(
                    type=RealtimeEventType.FUNCTION_CALL_DONE,
                    data={
                        "call_id": fc.id,
                        "name": fc.name,
                        "arguments": json.dumps(fc.args) if isinstance(fc.args, dict) else str(fc.args),
                    },
                ))

        # Tool call cancellation
        if hasattr(message, "tool_call_cancellation") and message.tool_call_cancellation:
            # Model cancelled pending function calls (e.g., due to interruption)
            logger.info("Tool call cancelled by model")

        return events
```

### Key Implementation Details

**Audio format**: Gemini Live accepts 16kHz PCM input and outputs 24kHz PCM. This matches our AudioIO contract exactly — no resampling needed between VoiceSession and the model.

**Message parsing**: The `_parse_message` method handles all Gemini Live server message types. The LiveKit plugin (`realtime_api.py`, 1368 lines) was the primary reference. Our implementation is simpler because we don't need to handle the LiveKit-specific concerns (room events, participant tracking, etc.).

**Session resumption**: Gemini Live sends `session_resumption_update` messages with opaque tokens. We store the latest token and pass it when creating a new session after disconnect. This gives seamless conversation continuity across WebRTC reconnects.

**Function calls**: Gemini sends `tool_call` messages with function name and arguments. We emit `FUNCTION_CALL_DONE` events. VoiceSession dispatches to the ToolBridge (Phase 5), which executes the tool and calls `send_function_result()` to feed the result back.

**Interruption**: Gemini detects user speech during model output and sends `server_content.interrupted = true`. We emit `INTERRUPTED` so VoiceSession clears the audio output buffer. No explicit interrupt signal needs to be sent to the model — it handles it from the audio stream.

**Error handling**: If the receive loop fails, we emit `SESSION_ERROR` and set `_connected = False`. VoiceSession can detect this and attempt reconnection with the resumption token.

### Acceptance Criteria

- [ ] `GeminiRealtimeModel.create_session()` returns a connected `GeminiRealtimeSession`
- [ ] `push_audio()` sends PCM data to Gemini without error
- [ ] `events()` yields `AUDIO_DELTA` events with 24kHz PCM audio
- [ ] `events()` yields `FUNCTION_CALL_DONE` events with call_id, name, arguments
- [ ] `send_function_result()` feeds results back to Gemini
- [ ] `generate_reply(text)` triggers a spoken response from Gemini
- [ ] Session resumption token is captured and usable for reconnection
- [ ] Interruption events are correctly parsed and emitted
- [ ] `close()` cleanly shuts down the WebSocket connection
- [ ] Input transcription events are emitted when Gemini transcribes user speech

### Estimated Size

~600 new lines across 3 files (model.py ~100, session.py ~450, __init__.py ~10)

---

## 9. Phase 5 — Tool Bridge (All Tools for P Worker)

**Why fifth**: With the Gemini backend working (Phase 4), we need to give it tools. **Both P and E Workers have access to ALL tools — no artificial curation.** Gemini decides what to call directly and what to delegate. Every tool execution (success or failure) is logged to the Task Ledger for traceability.

### New Files

```
src/aether/voice/backends/gemini/tool_bridge.py   — Tool declarations + execution + ledger logging
```

### Design

The Tool Bridge serves three purposes:
1. **Tool declarations** — Converts ALL tools from `ToolRegistry` to Gemini function calling format
2. **Tool execution** — When Gemini calls a tool, the bridge executes it via the existing tool instances
3. **Execution logging** — Every tool call (success or failure) is logged to the Task Ledger with the session_id

There is **no P-tool / E-tool split**. Gemini sees every tool that Claude sees. The only additional tool is `delegate_to_agent` (Phase 6) for when Gemini decides a task needs Claude's multi-step reasoning.

Gemini decides on its own whether to call a tool directly or delegate. The system prompt (PROMPT_P.md) gives guidance, but the tool set is identical.

```python
"""
Tool Bridge — exposes ALL tools to Gemini with execution logging.

No P-tool / E-tool split. Both P Worker (Gemini) and E Worker (Claude)
have access to the full ToolRegistry. Every tool execution — success or
failure — is logged to the Task Ledger for observability.

The bridge converts ToolRegistry tools to Gemini function declarations
and handles execution + logging when Gemini calls them.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether.session.ledger import TaskLedger
    from aether.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolBridge:
    """
    Bridges Gemini function calls to the existing ToolRegistry.
    
    Provides:
      1. Tool declarations in Gemini format (from ToolRegistry)
      2. Tool execution with Task Ledger logging
      3. Additional tools (delegate_to_agent) registered separately
    """

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        task_ledger: "TaskLedger",
        session_id: str = "voice",
    ) -> None:
        self._registry = tool_registry
        self._ledger = task_ledger
        self._session_id = session_id
        # Additional tools not in ToolRegistry (e.g., delegate_to_agent)
        self._extra_declarations: list[dict[str, Any]] = []
        self._extra_handlers: dict[str, Any] = {}

    def register_extra(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Any,
    ) -> None:
        """Register an additional tool not in ToolRegistry (e.g., delegate_to_agent)."""
        self._extra_declarations.append({
            "name": name,
            "description": description,
            "parameters": parameters,
        })
        self._extra_handlers[name] = handler

    def get_declarations(self) -> list[dict[str, Any]]:
        """Get ALL tool declarations in Gemini function calling format.
        
        Includes every tool from ToolRegistry + any extra tools.
        """
        declarations = []

        # All tools from ToolRegistry (built-in + plugin tools)
        for tool in self._registry.list_tools():
            schema = tool.to_openai_schema()
            # Convert OpenAI format → Gemini format
            func = schema.get("function", schema)
            declarations.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })

        # Extra tools (delegate_to_agent, etc.)
        declarations.extend(self._extra_declarations)

        return declarations

    async def execute(self, name: str, arguments: str, session_id: str | None = None) -> str:
        """Execute a tool and log the execution to the Task Ledger.
        
        Every execution — success or failure — is logged with:
        - session_id (for traceability / Langfuse correlation)
        - tool name, arguments, result, duration, error status
        
        Args:
            name: Tool name
            arguments: JSON-encoded arguments string
            session_id: Override session_id for this execution (default: bridge's session_id)
        
        Returns:
            JSON result string
        """
        effective_session_id = session_id or self._session_id
        args = json.loads(arguments) if arguments else {}
        start_time = time.monotonic()
        error = None
        result_str = ""

        try:
            # Check extra handlers first (delegate_to_agent, etc.)
            if name in self._extra_handlers:
                result_str = await self._extra_handlers[name](args)
                return result_str

            # Look up in ToolRegistry
            tool = self._registry.get_tool(name)
            if not tool:
                error = f"Unknown tool: {name}"
                result_str = json.dumps({"error": error})
                return result_str

            # Execute the tool
            result = await tool.execute(**args)
            if hasattr(result, "output"):
                result_str = result.output
                if hasattr(result, "error") and result.error:
                    error = result_str
            else:
                result_str = json.dumps(result) if isinstance(result, dict) else str(result)

            return result_str

        except Exception as e:
            error = str(e)
            result_str = json.dumps({"error": error})
            logger.exception("Tool %s failed", name)
            return result_str

        finally:
            # Log EVERY execution to the Task Ledger (success and failure)
            duration_ms = int((time.monotonic() - start_time) * 1000)
            try:
                task_id = await self._ledger.submit(
                    session_id=effective_session_id,
                    task_type="tool_call",
                    payload={
                        "tool_name": name,
                        "arguments": args,
                        "source": "p_worker",
                    },
                    priority="normal",
                )
                if error:
                    await self._ledger.set_error(task_id, error)
                else:
                    await self._ledger.set_complete(task_id, {
                        "result": result_str[:2000],  # Truncate large results
                        "duration_ms": duration_ms,
                    })
            except Exception:
                logger.warning("Failed to log tool execution to ledger", exc_info=True)
```

### No Tool Curation — Gemini Sees Everything

Both P and E Workers have access to the **full tool set**. There is no "P-tool" vs "E-tool" distinction.

| What Gemini can call directly | What Gemini delegates via `delegate_to_agent` |
|---|---|
| Any tool in ToolRegistry | Any task requiring multi-step reasoning |
| `world_time`, `web_search`, `save_memory`, etc. | Tasks needing multiple sequential tool calls |
| `gmail_read`, `calendar_list`, `spotify_play`, etc. | Tasks needing planning + execution |
| Even `read_file`, `run_command`, `write_file` | Tasks where Gemini's tool calling is unreliable |

**Gemini decides.** The PROMPT_P.md gives guidance on when to delegate, but the tool set is not artificially restricted. If Gemini can handle a Gmail read directly, it does. If it needs Claude to draft a multi-email sequence, it delegates.

### Task Ledger Tracks Everything

Every tool execution is logged to the Task Ledger, regardless of:
- **Who called it**: P Worker (Gemini) or E Worker (Claude)
- **Whether it succeeded or failed**: Both are logged
- **What type of tool**: Built-in, plugin, or delegation

This gives full observability:

```sql
-- "What tools were called in this voice session?"
SELECT * FROM tasks WHERE session_id = 'voice-abc123' ORDER BY submitted_at;

-- "What failed in the last hour?"
SELECT * FROM tasks WHERE status = 'error' AND submitted_at > datetime('now', '-1 hour');

-- "How long did Gmail calls take on average?"
SELECT AVG(json_extract(result, '$.duration_ms'))
FROM tasks WHERE json_extract(payload, '$.tool_name') = 'gmail_read';
```

### System Prompts: PROMPT_P.md and PROMPT_E.md

The P Worker (Gemini) and E Worker (Claude) each get their own system prompt file. This keeps personality and routing instructions separate and maintainable.

| File | Model | Purpose |
|---|---|---|
| `app/src/aether/PROMPT_P.md` | Gemini (P Worker) | Voice personality, tool routing, delegation rules, conversation style |
| `app/src/aether/PROMPT.md` (renamed to `PROMPT_E.md`) | Claude (E Worker) | Task execution, tool-calling strategy, reasoning approach |

**`PROMPT_P.md`** is loaded by the factory (Phase 7) and passed as `instructions` to `GeminiRealtimeModel.create_session()`. It contains:

1. **Identity and personality** — who Aether is, tone, warmth, conversational style (shared with E Worker but adapted for voice/realtime)
2. **Tool routing rules** — when to handle directly vs. when to delegate
3. **Voice-specific rules** — acknowledgment prefixes, no markdown, natural number spelling, concise responses
4. **Memory context injection point** — where retrieved facts/memories/decisions are inserted before each turn
5. **Delegation instructions** — how to use `delegate_to_agent`, what context to pass, how to communicate wait times to the user

Example content for `PROMPT_P.md`:

```markdown
# Aether — P Worker (Voice & Conversation)

You are Aether, a warm and capable personal assistant. You are the user's
always-on companion — available via voice, text, and video.

## Your tools

You have access to ALL tools — the same full set that the agent (Claude) has.
This includes lookups (time, weather, search), memory (save/recall), email,
calendar, file operations, code execution, music control, and more.

For most requests, call the tool directly. For complex tasks — multi-step
reasoning, long-form writing, research synthesis, or anything requiring
sequential tool chains — use `delegate_to_agent`. The agent (Claude) runs
an autonomous loop and handles it end-to-end.

## When to delegate

Delegate when the task requires:
- Multiple tool calls in sequence (e.g., read 3 emails then summarize)
- Planning + execution (e.g., draft a reply considering calendar conflicts)
- Long-form content generation (e.g., write a report, draft a document)
- Research that needs synthesis across multiple sources
- Any task that takes more than a few seconds of reasoning

## When to handle directly

Handle directly when:
- A single tool call answers the question (time, weather, search, memory)
- The user wants to save or recall a memory
- The user wants to control music or set a reminder
- The request is conversational (greeting, small talk, quick question)

## Voice rules

- Start EVERY response with a short acknowledgment: "Sure.", "Got it.", "Okay."
- Never use markdown, bullet points, or code blocks
- Spell out numbers naturally: "twenty three" not "23"
- Keep responses under 3 sentences unless asked for detail
- No emoji, no special characters
- When delegating, tell the user: "Let me work on that" or "Give me a moment"
- When a delegation completes, summarize the result conversationally

## Memory

Before each turn, you receive relevant facts, memories, and decisions about
this user. Use them naturally — don't announce that you're using memory.
```

**`PROMPT_E.md`** (the existing `PROMPT.md`, renamed for clarity) stays as-is — it's Claude's task execution prompt. The E Worker never talks to the user directly; it receives structured task descriptions and returns structured results.

**Loading at startup** (in `factory.py` / `main.py`):

```python
def _load_p_prompt() -> str:
    """Load P Worker system prompt from PROMPT_P.md."""
    prompt_path = Path(__file__).parent / "PROMPT_P.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "You are Aether, a warm and helpful personal assistant."
```

### Acceptance Criteria

- [ ] `ToolBridge` converts all `ToolRegistry` tools to Gemini function declarations
- [ ] `get_declarations()` returns valid Gemini function calling format
- [ ] ALL tools from ToolRegistry are exposed (no artificial curation)
- [ ] `delegate_to_agent` is registered as an extra tool
- [ ] `execute()` dispatches to correct tool and returns JSON result
- [ ] Every execution (success and failure) is logged to Task Ledger
- [ ] Unknown tool calls return a clear JSON error
- [ ] Plugin tools are included when the plugin is loaded

### Estimated Size

~200 new lines in 1 file

---

## 10. Phase 6 — Delegation Bridge (P→E)

**Why sixth**: With the ToolBridge working (Phase 5), we need the escape hatch — `delegate_to_agent` — for when Gemini encounters a request too complex for direct tool calls. This is the P Worker → E Worker bridge.

### New Files

```
src/aether/voice/backends/gemini/bridge.py   — delegate_to_agent() implementation
```

### Design

The delegation bridge is a special tool that Gemini calls when a request needs Claude's reasoning. It:

1. Submits a task to the Task Ledger
2. Waits for the E Worker to complete it (with timeout)
3. Returns the result to Gemini, which speaks it to the user

```python
"""
Delegation Bridge — P Worker → E Worker via Task Ledger.

When Gemini encounters a complex request, it calls delegate_to_agent().
This function:
  1. Writes a task to the Task Ledger (P Worker side)
  2. Waits for the E Worker (Claude) to complete it
  3. Returns the result as a string for Gemini to speak

The E Worker picks up the task via its normal SessionLoop, runs Claude
with full tool access, and writes the result back to the Task Ledger.

Communication is entirely via the Task Ledger — no direct function calls
between P and E Workers. This maintains the clean P/E separation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether.agent import AgentCore
    from aether.session.ledger import TaskLedger

logger = logging.getLogger(__name__)

# Maximum time to wait for E Worker to complete a delegated task
DELEGATION_TIMEOUT_S = 120  # 2 minutes
DELEGATION_POLL_INTERVAL_S = 0.5  # Check every 500ms


class DelegationBridge:
    """
    Bridges P Worker (Gemini) to E Worker (Claude) via Task Ledger.
    
    Registered as the `delegate_to_agent` extra tool in ToolBridge.
    """

    def __init__(
        self,
        agent_core: "AgentCore",
        task_ledger: "TaskLedger",
        voice_session_id: str = "voice",
    ) -> None:
        self._agent = agent_core
        self._ledger = task_ledger
        self._voice_session_id = voice_session_id

    async def delegate(self, args: dict[str, Any]) -> str:
        """
        Handle a delegate_to_agent function call from Gemini.
        
        Args (from Gemini):
            task: str — description of what needs to be done
            context: str — any additional context from the conversation
        
        Returns:
            JSON string with the E Worker's result
        """
        task_description = args.get("task", "")
        context = args.get("context", "")

        if not task_description:
            return json.dumps({"error": "No task description provided"})

        logger.info("Delegating to E Worker: %s", task_description[:100])

        # Each delegation gets its own session ID so conversation history
        # doesn't bleed between unrelated tasks. The voice session ID is
        # used as a parent reference for traceability.
        delegation_id = str(uuid.uuid4())[:8]
        e_session_id = f"delegation-{delegation_id}"

        # 1. Submit task to the Task Ledger
        prompt = task_description
        if context:
            prompt = f"{task_description}\n\nContext: {context}"

        task_id = await self._ledger.submit(
            session_id=e_session_id,
            task_type="sub_agent",
            payload={
                "action": "run_session",
                "prompt": prompt,
                "source": "p_worker_delegation",
                "parent_voice_session": self._voice_session_id,
            },
            priority="high",
        )

        # 2. Trigger E Worker — run_session is the full autonomous loop
        #    with tool calling, multi-step reasoning, and sub-agent spawning.
        #    generate_reply is single-turn only and cannot do multi-step work.
        e_worker_task = asyncio.create_task(
            self._run_e_worker(task_id, e_session_id, prompt)
        )

        # 3. Wait for completion (with timeout)
        try:
            result = await asyncio.wait_for(
                self._wait_for_completion(task_id),
                timeout=DELEGATION_TIMEOUT_S,
            )
            return json.dumps({"result": result})
        except asyncio.TimeoutError:
            logger.warning("Delegation timed out after %ds: %s", DELEGATION_TIMEOUT_S, task_description[:100])
            # Don't cancel — let the E Worker finish in the background.
            # The notification system will deliver the result when ready.
            return json.dumps({
                "result": "I'm still working on that. I'll let you know when it's done.",
                "status": "timeout",
                "task_id": task_id,
            })

    async def _run_e_worker(self, task_id: str, session_id: str, prompt: str) -> None:
        """Run the E Worker for a delegated task.
        
        Uses AgentCore.run_session() — the full autonomous SessionLoop with
        tool calling, multi-step reasoning, and sub-agent spawning. This is
        NOT generate_reply() which is single-turn only.
        """
        try:
            await self._ledger.set_running(task_id)
            result = await self._agent.run_session(
                session_id=session_id,
                user_message=prompt,
                background=False,  # Run to completion, don't return immediately
            )
            await self._ledger.set_complete(task_id, {"result": result})
        except Exception as e:
            logger.exception("E Worker failed for task %s", task_id)
            await self._ledger.set_error(task_id, str(e))

    async def _wait_for_completion(self, task_id: str) -> str:
        """Poll the Task Ledger until the task is complete."""
        while True:
            task = await self._ledger.get_task(task_id)
            if task.status.value == "complete":
                result = task.result or {}
                return result.get("result", str(result))
            elif task.status.value == "error":
                return f"I encountered an error: {task.error or 'unknown error'}"
            await asyncio.sleep(DELEGATION_POLL_INTERVAL_S)

    # Constants for ToolBridge registration
    DESCRIPTION = (
        "Delegate a complex task to the reasoning agent (Claude). "
        "Use this for tasks that require multi-step reasoning, writing long content, "
        "managing email threads, calendar scheduling with conflict resolution, "
        "research synthesis, or any task that needs careful planning and "
        "multiple sequential tool calls. The agent runs an autonomous loop "
        "with full tool access and handles it end-to-end."
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Clear description of what needs to be done",
            },
            "context": {
                "type": "string",
                "description": "Any relevant context from the conversation",
            },
        },
        "required": ["task"],
    }

    async def execute(self, args: dict) -> str:
        """Entry point for ToolBridge — wraps self.delegate()."""
        return await self.delegate(
            task=args["task"],
            context=args.get("context"),
        )
```

### Delegation Flow

```
User: "Draft an email to John about the project update"
  │
  ▼
Gemini Live (P Worker)
  │ Recognizes this needs multi-step reasoning
  │ Calls delegate_to_agent(task="Draft an email to John about the project update")
  │
  ▼
DelegationBridge.delegate()
  │ 1. Writes task to Task Ledger (status: pending)
  │ 2. Starts E Worker coroutine
  │ 3. Polls Task Ledger for completion
  │
  ▼
SessionLoop.run_session() (E Worker / Claude — full autonomous loop)
  │ Claude reasons about the task
  │ Calls gmail_create_draft tool
  │ Calls gmail_send tool
  │ Returns: "I've drafted and sent the email to John about the project update."
  │
  ▼
Task Ledger updated (status: complete, result: "I've drafted and sent...")
  │
  ▼
DelegationBridge returns result to Gemini
  │
  ▼
Gemini speaks: "I've drafted and sent the email to John about the project update."
  │
  ▼
User hears the response
```

### Timeout Handling

If the E Worker takes longer than 2 minutes (configurable), the bridge returns a partial response: "I'm still working on that. I'll let you know when it's done." The E Worker continues running in the background. When it completes, the result is stored in the Task Ledger. The notification system (already implemented) can deliver the result when ready.

### Acceptance Criteria

- [ ] `delegate_to_agent` is registered as an extra tool in ToolBridge
- [ ] Delegation writes a task to the Task Ledger
- [ ] E Worker (Claude) picks up and executes the task
- [ ] Result is returned to Gemini for spoken delivery
- [ ] Timeout returns a graceful partial response
- [ ] E Worker errors are caught and returned as readable messages
- [ ] Task Ledger tracks the full lifecycle (pending → running → complete/error)

### Estimated Size

~200 new lines in 1 file

---

## 11. Phase 7 — Factory + Wiring

**Why last**: Everything is built. This phase wires it all together — config, factory, main.py startup.

### New Files

```
src/aether/voice/factory.py   — Creates RealtimeModel from config
```

### Modified Files

```
src/aether/core/config.py     — Add VoiceBackendConfig
src/aether/main.py            — Wire RealtimeModel into transport creation
```

### `config.py` — VoiceBackendConfig

```python
@dataclass(frozen=True)
class VoiceBackendConfig:
    """Voice backend configuration.
    
    Controls which realtime model powers the voice pipeline.
    Swapping backends requires only changing AETHER_VOICE_BACKEND.
    """
    backend: str = "gemini"           # "gemini" | "openai" | "classic" (future)
    # Gemini-specific
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-preview-native-audio"
    gemini_voice: str = "Puck"        # Aoede, Charon, Fenrir, Kore, Puck
    # Shared
    instructions: str = ""            # Override system instructions (default: from PROMPT.md)
    temperature: float = 0.7
    enable_session_resumption: bool = True

    @classmethod
    def from_env(cls) -> "VoiceBackendConfig":
        return cls(
            backend=os.getenv("AETHER_VOICE_BACKEND", "gemini"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("AETHER_GEMINI_MODEL", "gemini-2.5-flash-preview-native-audio"),
            gemini_voice=os.getenv("AETHER_GEMINI_VOICE", "Puck"),
            instructions=os.getenv("AETHER_VOICE_INSTRUCTIONS", ""),
            temperature=float(os.getenv("AETHER_VOICE_TEMPERATURE", "0.7")),
            enable_session_resumption=os.getenv("AETHER_VOICE_SESSION_RESUMPTION", "true").lower() == "true",
        )
```

**AetherConfig update**:
```python
@dataclass(frozen=True)
class AetherConfig:
    # ... existing fields ...
    voice_backend: VoiceBackendConfig = field(default_factory=VoiceBackendConfig)

    @classmethod
    def from_env(cls) -> AetherConfig:
        return cls(
            # ... existing ...
            voice_backend=VoiceBackendConfig.from_env(),
        )
```

### `factory.py` — RealtimeModel Factory

```python
"""
Voice backend factory — creates RealtimeModel from config.

Swapping backends requires only changing AETHER_VOICE_BACKEND env var.
No transport code changes needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aether.core.config import config
from aether.voice.realtime import RealtimeModel, RealtimeModelConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def create_realtime_model() -> RealtimeModel:
    """Create a RealtimeModel based on config.
    
    Returns:
        A RealtimeModel instance ready to create sessions.
    
    Raises:
        ValueError: If the configured backend is not supported.
    """
    backend = config.voice_backend.backend

    if backend == "gemini":
        from aether.voice.backends.gemini import GeminiRealtimeModel

        model_config = RealtimeModelConfig(
            model=config.voice_backend.gemini_model,
            api_key=config.voice_backend.gemini_api_key,
            voice=config.voice_backend.gemini_voice,
            instructions=config.voice_backend.instructions,
            temperature=config.voice_backend.temperature,
            enable_session_resumption=config.voice_backend.enable_session_resumption,
        )
        logger.info("Voice backend: Gemini Live (%s)", model_config.model)
        return GeminiRealtimeModel(model_config)

    # Future backends:
    # elif backend == "openai":
    #     from aether.voice.backends.openai import OpenAIRealtimeModel
    #     ...
    # elif backend == "classic":
    #     from aether.voice.backends.classic import ClassicPipelineModel
    #     ...

    raise ValueError(f"Unsupported voice backend: {backend}")
```

### `main.py` — Wiring Changes

The key changes in `main.py`:

```python
# BEFORE (current):
from aether.voice.webrtc import WebRTCVoiceTransport
tts_provider = build_tts_provider()
webrtc_transport = WebRTCVoiceTransport(agent=agent_core, tts_provider=tts_provider)
# Text API goes directly to Claude via AgentCore

# AFTER (new):
from aether.voice.factory import create_realtime_model
from aether.voice.backends.gemini.tool_bridge import ToolBridge
from aether.voice.backends.gemini.bridge import DelegationBridge
from aether.voice.webrtc import WebRTCVoiceTransport

# Create realtime model (the P Worker)
realtime_model = create_realtime_model()

# Build ToolBridge — exposes ALL tools from ToolRegistry to Gemini
tool_bridge = ToolBridge(
    tool_registry=tool_registry,
    task_ledger=task_ledger,
)

# Build delegation bridge and register as extra tool
bridge = DelegationBridge(
    agent_core=agent_core,
    task_ledger=task_ledger,
)
tool_bridge.register_extra(
    name="delegate_to_agent",
    description=bridge.DESCRIPTION,
    parameters=bridge.PARAMETERS,
    handler=bridge.execute,
)

# Load P Worker system prompt
p_instructions = _load_p_prompt()

# Create a shared RealtimeSession for text-only connections.
# Voice connections each get their own session (via VoiceSession).
# Text connections share one persistent session per user.
text_session = await realtime_model.create_session(
    instructions=p_instructions,
    tools=tool_bridge.get_declarations(),
)

# Create voice transport (WebRTC)
webrtc_transport = WebRTCVoiceTransport(
    realtime_model=realtime_model,
    tool_bridge=tool_bridge,
    instructions=p_instructions,
)

# Text API now routes through Gemini (P Worker)
# POST /v1/chat/completions → text_session.generate_reply(text)
# POST /chat → text_session.generate_reply(text)
# The response is Gemini's text output (possibly after delegation to Claude)
```

**Text API routing change:**

```python
# BEFORE: POST /chat → AgentCore.generate_reply() → Claude directly
# AFTER:  POST /chat → text_session.generate_reply() → Gemini (P Worker)
#         Gemini handles directly OR delegates to Claude via bridge

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_message = _extract_user_message(body)
    session_id = f"http-{body.get('user_id', 'anon')}"

    # Route through P Worker (Gemini)
    await text_session.generate_reply(user_message)

    # Collect text response from Gemini's event stream
    response_text = ""
    async for event in text_session.events():
        if event.type == RealtimeEventType.TEXT_DELTA:
            response_text += event.data.get("text", "")
        elif event.type == RealtimeEventType.TURN_DONE:
            break

    return StreamingResponse(...)  # Stream the response
```

**Function call wiring** — VoiceSession's `on_function_call` handler:

```python
# In WebRTCVoiceTransport, when creating VoiceSession:
async def _handle_function_call(call_id: str, name: str, arguments: str) -> str:
    """Route function calls to ToolBridge."""
    return await tool_bridge.execute(name, arguments)

vs = VoiceSession(
    session_id=session_id,
    realtime_model=realtime_model,
    on_function_call=_handle_function_call,
    instructions=voice_instructions,
    tools=tool_bridge.get_declarations(),
)
```

### Environment Variables (New)

| Variable | Default | Description |
|---|---|---|
| `AETHER_VOICE_BACKEND` | `gemini` | Which realtime model backend to use |
| `GEMINI_API_KEY` | (required) | Google AI API key for Gemini |
| `AETHER_GEMINI_MODEL` | `gemini-2.5-flash-preview-native-audio` | Gemini model name |
| `AETHER_GEMINI_VOICE` | `Puck` | Gemini voice (Aoede, Charon, Fenrir, Kore, Puck) |
| `AETHER_VOICE_INSTRUCTIONS` | (from PROMPT.md) | Override system instructions |
| `AETHER_VOICE_TEMPERATURE` | `0.7` | Model temperature |
| `AETHER_VOICE_SESSION_RESUMPTION` | `true` | Enable session resumption |

### What Gets Removed / Changed in main.py

- `tts_provider = build_tts_provider()` — no more TTS provider for voice (Gemini outputs audio directly)
- `WebRTCVoiceTransport(agent=agent_core, tts_provider=tts_provider)` — replaced with new constructor
- Any direct `agent_core._voice_transport` wiring for spoken notifications — replaced by `VoiceSession.inject_text()`
- `POST /v1/chat/completions` and `POST /chat` — no longer call `AgentCore.generate_reply()` directly. They route through the P Worker's `text_session.generate_reply()` instead.
- The OpenAI-compatible streaming format is preserved — the HTTP handler collects Gemini's text events and formats them as SSE chunks in the same format as before. Clients don't notice the change.

### Acceptance Criteria

- [ ] `AETHER_VOICE_BACKEND=gemini` creates a `GeminiRealtimeModel`
- [ ] `VoiceBackendConfig` is part of `AetherConfig` and loads from env
- [ ] `create_realtime_model()` returns a working model instance
- [ ] WebRTC transport creates VoiceSession with RealtimeModel (not AgentCore + TTS)
- [ ] Telephony transport creates VoiceSession with RealtimeModel (not AgentCore + TTS)
- [ ] ToolBridge is wired and all tools are available to Gemini
- [ ] Delegation bridge is wired and functional
- [ ] Spoken notifications use `inject_text()` instead of direct TTS
- [ ] All existing WebRTC signaling endpoints still work
- [ ] All existing telephony WebSocket handling still works

### Estimated Size

~80 new lines (factory.py), ~60 modified lines (config.py + main.py)

---

## 12. Summary

### Size Estimate

| Phase | New Lines | Modified Lines | Deleted Lines | Net Change |
|---|---|---|---|---|
| 1 — AudioIO + RealtimeModel ABCs | ~200 | 0 | 0 | +200 |
| 2 — VoiceSession Rewrite | 0 | ~900 (rewrite) | ~208 (turn_detection.py) | -700 |
| 3 — Transport Update | ~30 | ~120 | 0 | +150 |
| 4 — Gemini Realtime Backend | ~600 | 0 | 0 | +600 |
| 5 — P-Tools Registry | ~250 | 0 | 0 | +250 |
| 6 — Delegation Bridge | ~200 | 0 | 0 | +200 |
| 7 — Factory + Wiring | ~80 | ~60 | ~20 | +120 |
| **Total** | **~1,360** | **~1,080** | **~228** | **+820** |

**Files**: 8 new, 5 modified, 1 deleted

### What Aether Can Do After Each Phase

| After Phase | New Capability |
|---|---|
| Phase 1 | Clean ABCs defined — all subsequent work has stable interfaces to build against |
| Phase 2 | VoiceSession is a thin orchestrator — ready for any realtime model backend |
| Phase 3 | Transports use AudioIO protocol — fully decoupled from model backend |
| Phase 4 | Gemini Live is connected — audio in/out works, function calls work |
| Phase 5 | Gemini can handle simple requests directly — time, weather, memory, search |
| Phase 6 | Complex requests delegate to Claude — email, calendar, research, code |
| Phase 7 | Everything wired together — config-driven backend selection, production ready |

### Architecture After All Phases

**All user communication flows through Gemini (P Worker).** Claude is E-only.

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Communication                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   WebRTC     │  │  Telephony   │  │  REST/WS     │          │
│  │  (voice)     │  │  (voice)     │  │  (text)      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                   │
│    AudioIO Protocol        │         generate_reply(text)        │
│         │                  │                  │                   │
│         └──────────────────┼──────────────────┘                  │
│                            │                                     │
│                     ALL → Gemini (P Worker)                      │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│              ┌───────────────────────────────┐                   │
│              │    Gemini Live Backend         │                   │
│              │  (GeminiRealtimeSession)       │                   │
│              │                                │                   │
│              │  Voice: native audio in/out    │                   │
│              │  Text: generate_reply(text)    │                   │
│              │  Video: native multimodal      │                   │
│              │                                │                   │
│              │  ┌──────────┐ ┌─────────┐     │                   │
│              │  │ P-Tools  │ │ Bridge  │     │                   │
│              │  │ Registry │ │ (P→E)   │     │                   │
│              │  └────┬─────┘ └────┬────┘     │                   │
│              └───────┼────────────┼──────────┘                   │
│                      │            │                               │
│         ┌────────────┘            └────────────┐                 │
│         ▼                                      ▼                 │
│   Simple tools (60%)                    Task Ledger              │
│   - world_time                               │                   │
│   - weather                                  ▼                   │
│   - search_memory                     E Worker (Claude)          │
│   - web_search                        - Gmail                    │
│   - spotify                           - Calendar                 │
│   - wikipedia                         - Drive                    │
│   - ...                               - Sub-agents              │
│                                       - Research                 │
│                                       - Code gen                 │
│                                                                  │
│   ┌──────────────────────────────────────────────────┐          │
│   │  Memory Sync (runs after every P Worker turn)     │          │
│   │  Transcript → SessionStore → Memory Extraction    │          │
│   └──────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

### Four Scenarios After All Phases

**Scenario 1 — Simple voice request (60% of voice traffic)**
User says "What time is it in Tokyo?" → Gemini Live receives audio → recognizes speech → calls `world_time` via ToolBridge → gets result → speaks "It's 3:47 PM in Tokyo" → ~500ms total latency. Claude is never involved.

**Scenario 2 — Complex voice request (40% of voice traffic)**
User says "Draft an email to John about the project update" → Gemini Live receives audio → recognizes this needs reasoning → calls `delegate_to_agent` → Task Ledger → Claude (E Worker) runs full SessionLoop, drafts email via Gmail tools → result returns to Gemini → Gemini speaks "I've drafted and sent the email to John" → ~5-15s depending on complexity.

**Scenario 3 — Text chat (REST/WebSocket)**
User types "What's on my calendar today?" via dashboard → `text_session.generate_reply(text)` → Gemini processes → calls `delegate_to_agent` (calendar requires Google Calendar API) → Claude fetches calendar → result returns → Gemini responds as text → streamed to client. Same P→E flow, just text instead of audio.

**Scenario 4 — Multi-modal (voice + video)**
User shows their screen and says "What's wrong with this code?" → Gemini Live receives audio + video → analyzes both → either answers directly (simple syntax error) or delegates to Claude (complex architectural issue). Same pipeline, same ABCs — video is just another input modality that Gemini handles natively.

### Backend Swappability

To swap from Gemini to OpenAI Realtime:

1. Implement `OpenAIRealtimeModel` + `OpenAIRealtimeSession` in `backends/openai/`
2. Add `elif backend == "openai"` to `factory.py`
3. Set `AETHER_VOICE_BACKEND=openai`
4. Zero transport code changes. Zero VoiceSession changes. Zero ToolBridge changes.

To swap back to classic STT→LLM→TTS:

1. Implement `ClassicPipelineModel` + `ClassicPipelineSession` in `backends/classic/` (wraps Deepgram + Claude + TTS as a RealtimeSession)
2. Add `elif backend == "classic"` to `factory.py`
3. Set `AETHER_VOICE_BACKEND=classic`
4. Same — zero changes elsewhere.

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `google-genai` | ≥1.0.0 | Gemini Live API client |

No other new dependencies. The existing `aiortc`, `numpy`, `onnxruntime` dependencies remain for WebRTC and VAD.

### Migration Notes

- **No gradual migration.** VoiceSession internals are fully replaced. The old STT→LLM→TTS pipeline is deleted, not kept as a fallback. If Gemini Live has issues, the fix is to implement a `ClassicPipelineModel` backend, not to maintain two code paths.
- **Text API routes through Gemini.** `POST /v1/chat/completions` and `POST /chat` no longer go directly to Claude. They route through the P Worker (Gemini) via `text_session.generate_reply(text)`. The HTTP response format (SSE streaming) is preserved — clients don't notice the change. Claude is only reachable via the delegation bridge.
- **Existing tests.** Tests that reference VoiceSession internals (STT, EOU, turn detection) will need to be rewritten to test the new thin orchestrator. Tests for the E Worker (AgentCore, SessionLoop, tools) are unchanged. Text API tests will need updating to mock the Gemini session instead of AgentCore directly.

---

### Conversation Memory Sync (Day One Requirement)

When Gemini handles a request directly via ToolBridge (the 60% path), that interaction never touches Claude or AgentCore. Without explicit sync, the agent "forgets" everything that happened in the P Worker path — facts mentioned, memories saved, decisions made. This is unacceptable. Memory sync must work from day one.

**The problem:**
- User tells Gemini "Remember that my dentist appointment is Thursday" → Gemini calls `save_memory` via ToolBridge → memory is saved ✅
- But the conversation transcript (what the user said, what Gemini replied) is NOT persisted to `SessionStore` → no memory extraction happens → the agent doesn't learn patterns from P Worker conversations ✗
- User asks Claude (via delegation) "What did we talk about earlier?" → Claude has no record of the P Worker conversation ✗

**The solution: P Worker Transcript Logger**

After every Gemini turn (user speech + model response), the P Worker logs the transcript to `SessionStore` and triggers async memory extraction — exactly like the E Worker does today.

```
Gemini turn completes
    │
    ├── 1. Log transcript to SessionStore
    │      user_message: input transcription (from INPUT_SPEECH_TRANSCRIPTION_COMPLETED)
    │      assistant_message: output transcription (from TEXT_DONE)
    │      session_id: voice session ID
    │
    ├── 2. Trigger memory extraction (async, non-blocking)
    │      Submit to Task Ledger: type=memory_extract
    │      E Worker picks it up, extracts facts/memories/decisions
    │
    └── 3. Continue (don't wait for extraction)
```

**Implementation — in VoiceSession._handle_event():**

```python
case RealtimeEventType.INPUT_SPEECH_TRANSCRIPTION_COMPLETED:
    transcript = event.data.get("transcript", "")
    if transcript:
        self._pending_user_transcript = transcript

case RealtimeEventType.TEXT_DONE:
    # Model finished its text response for this turn
    model_text = self._accumulated_model_text
    user_text = self._pending_user_transcript

    if user_text and model_text:
        # 1. Persist to SessionStore
        await self._session_store.ensure_session(self.session_id)
        await self._session_store.append_user_message(self.session_id, user_text)
        await self._session_store.append_assistant_message(self.session_id, model_text)

        # 2. Trigger async memory extraction via Task Ledger
        await self._task_ledger.submit(
            session_id=self.session_id,
            task_type="memory_extract",
            payload={
                "user_message": user_text,
                "assistant_message": model_text,
            },
            priority="background",
        )

    # Reset for next turn
    self._pending_user_transcript = ""
    self._accumulated_model_text = ""
```

**What this gives you:**
1. **Full conversation history** — every P Worker turn is in `SessionStore`, queryable by Claude, visible in dashboard
2. **Memory extraction** — facts, memories, and decisions are extracted from P Worker conversations, same as E Worker
3. **Session continuity** — if the user switches from voice to text (or vice versa), the conversation history is continuous
4. **Nightly analysis** — P Worker conversations are included in the nightly memory analysis batch

**Dependencies:**
- `VoiceSession` needs access to `SessionStore` and `TaskLedger` (passed via constructor or factory)
- Gemini must have input/output transcription enabled (`input_audio_transcription` and `output_audio_transcription` in `LiveConnectConfig`) — already configured in Phase 4

**Memory retrieval before P Worker turns:**

For Gemini to use the user's memory (facts, decisions, etc.), the P Worker must inject relevant memory into Gemini's context before each turn. This is done via the system instructions update mechanism:

```python
# Before each turn (or periodically), retrieve relevant memory
# and update Gemini's instructions with the memory context
memory_context = await memory_store.search(user_transcript, limit=10)
decisions = await memory_store.get_decisions(active_only=True)

# Inject as a context update to the Gemini session
updated_instructions = f"{base_instructions}\n\n## Current context\n{memory_context}\n\n## Your rules\n{decisions}"
await realtime_session.update_instructions(updated_instructions)
```

This ensures Gemini has the same memory awareness as Claude — it knows the user's facts, remembers past conversations, and follows learned decisions.

**Phase impact:** This adds ~50 lines to VoiceSession (Phase 2) and requires `SessionStore` + `TaskLedger` to be passed to VoiceSession (Phase 7 wiring). No new files needed.

---

### Session ID as Universal Trace Key

Every interaction in Aether — voice, text, delegation, sub-agent — gets a unique `session_id`. This is the universal correlation key for tracing, debugging, and observability.

**Session ID assignment:**

| Context | Format | Example |
|---|---|---|
| Voice connection | `voice-{uuid[:8]}` | `voice-a1b2c3d4` |
| Text connection (HTTP) | `http-{user_id}` | `http-anon` |
| Text connection (WebSocket) | `ws-{uuid[:8]}` | `ws-e5f6g7h8` |
| Delegation (P→E) | `delegation-{uuid[:8]}` | `delegation-i9j0k1l2` |
| Sub-agent spawn | `sub-{uuid[:8]}` | `sub-m3n4o5p6` |

**Parent tracking:** Delegations and sub-agents carry a `parent_session_id` so the full call tree is reconstructable:

```
voice-a1b2c3d4                          ← user's voice session
  └── delegation-i9j0k1l2              ← Gemini delegated to Claude
        ├── (tool calls logged here)
        └── sub-m3n4o5p6               ← Claude spawned a sub-agent
              └── (tool calls logged here)
```

**What gets tagged with session_id:**

- Every `TaskLedger` entry (`submit()` requires `session_id`)
- Every tool execution via ToolBridge (`execute()` logs with `session_id`)
- Every E Worker tool execution (SessionLoop already tags with `session_id`)
- Every memory extraction task (`memory_extract` type in TaskLedger)
- Every `SessionStore` conversation turn (`append_user_message()` / `append_assistant_message()`)

**Observability integration (Langfuse / OpenTelemetry):**

The `session_id` maps directly to a Langfuse `trace_id` or OpenTelemetry `trace_id`. When we add observability (future phase), the wiring is:

```python
# In ToolBridge.execute() — already logs to TaskLedger
# Future: also emit a Langfuse span
langfuse.span(
    trace_id=session_id,       # ← session_id IS the trace_id
    name=f"tool:{tool_name}",
    input=arguments,
    output=result,
)

# In DelegationBridge.delegate() — already logs to TaskLedger
# Future: also start a Langfuse generation
langfuse.generation(
    trace_id=parent_session_id,
    name=f"delegation:{delegation_session_id}",
    model="claude-opus",
    input=task_description,
    output=result,
)
```

Because `session_id` is already threaded through every component (TaskLedger, ToolBridge, DelegationBridge, SessionStore, SessionLoop), adding Langfuse/OpenTelemetry is a thin wrapper — no architectural changes needed.

**Phase impact:** No new files. Session ID generation is in VoiceSession (Phase 2), DelegationBridge (Phase 6), and factory wiring (Phase 7). The pattern is already established in the E Worker (`SessionLoop` uses `session_id` throughout).

---

### TaskLedger — Expanded Scope

The TaskLedger was originally designed for E Worker task tracking (delegations, sub-agents). With the ToolBridge, its scope expands to track **all activity** across both workers.

**What TaskLedger now tracks:**

| Activity | Task Type | Source | When |
|---|---|---|---|
| P Worker tool execution | `tool_call` | `p_worker` | Every Gemini tool call via ToolBridge |
| E Worker tool execution | `tool_call` | `e_worker` | Every Claude tool call via ToolOrchestrator |
| Delegation (P→E) | `delegation` | `p_worker` | Gemini calls `delegate_to_agent` |
| Sub-agent spawn | `sub_agent` | `e_worker` | Claude spawns a sub-agent |
| Memory extraction | `memory_extract` | `p_worker` or `e_worker` | After conversation turns |

**Re-adding `tool_call` to TaskType enum:**

The `TaskType.TOOL_CALL` enum value was removed during the legacy fallback cleanup (commit `2458eec`) because it was unused at the time. The ToolBridge now needs it for logging individual tool executions. It must be re-added:

```python
# In app/src/aether/session/models.py
class TaskType(str, Enum):
    """Types of tasks that can be submitted to the Task Ledger."""
    GENERAL = "general"
    RESEARCH = "research"
    DELEGATION = "delegation"
    SUB_AGENT = "sub_agent"
    TOOL_CALL = "tool_call"          # ← re-add: individual tool executions
    MEMORY_EXTRACT = "memory_extract" # ← new: async memory extraction tasks
```

**Why this matters:**

1. **Full observability** — "What happened in this session?" is a single query: `SELECT * FROM tasks WHERE session_id = ? ORDER BY submitted_at`
2. **Performance tracking** — Tool execution times are logged, enabling latency analysis per tool, per worker
3. **Error forensics** — Failed tool calls are logged with error details, making debugging straightforward
4. **Audit trail** — Every action taken on behalf of the user is recorded, regardless of which worker performed it

**Phase impact:** `TaskType.TOOL_CALL` re-added in Phase 5 (ToolBridge). `TaskType.MEMORY_EXTRACT` added in Phase 2 (Conversation Memory Sync). No new files — these are enum additions to the existing `models.py`.

---

### Relationship to `implementation_plan.md`

This plan completes the P Worker half of the architecture. Together:

| Document | Scope | Status |
|---|---|---|
| `implementation_plan.md` | E Worker: Task Ledger, Session Loop, Sub-Agents, Compaction, Agent Types | ✅ All 7 phases complete |
| `voice_session_plan.md` (this) | P Worker: Gemini Live, AudioIO, RealtimeModel, ToolBridge, DelegationBridge | Pending (7 phases) |

After both plans are implemented, Aether has the full P/E Worker architecture from `Requirements.md` §2.2:
- **P Worker** (Gemini Live): always-on, multi-modal, handles 60% directly, delegates 40%
- **E Worker** (Claude Opus): autonomous session loop, full tool access, sub-agents, memory extraction
- **Task Ledger**: single communication channel between P and E, SQLite-backed, persistent, traceable

---

*This document is a living specification.*
