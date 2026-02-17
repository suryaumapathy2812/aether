# Project Aether v0.01 — The Bootloader

**Objective:** A working voice conversation with memory and vision. End-to-end. No polish, no framework, just a pipeline that works.

---

## What It Does

A user opens a web page, clicks a mic button, speaks. The system transcribes their speech, retrieves relevant memories, sends everything to an LLM, converts the response to speech, and plays it back. The user can also send images alongside voice. The system remembers conversations across sessions.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Web Client                     │
│  (mic input, audio playback, image capture)      │
│              WebSocket connection                 │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Pipeline Runner                     │
│                                                  │
│  AudioIn → STT → MemoryRetriever → LLM → TTS → AudioOut
│                                     ↑            │
│                               VisionFrame        │
└──────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Memory Store                        │
│  (SQLite + vector embeddings)                    │
└──────────────────────────────────────────────────┘
```

## Components to Build

### 1. Frame (data model)

Simple dataclass. Every piece of data flowing through the pipeline is a Frame.

```
Frame:
  id: str (uuid)
  type: "audio" | "text" | "vision" | "memory" | "control"
  data: bytes | str | dict
  metadata: dict
  timestamp: float
```

### 2. Processor (interface)

Every pipeline stage implements this. Nothing more.

```
Processor:
  name: str
  async start() → None
  async process(frame: Frame) → AsyncGenerator[Frame]
  async stop() → None
```

### 3. Pipeline Runner

Takes a list of Processors. Feeds frames through them in order. Handles async flow.

```
Pipeline:
  processors: list[Processor]
  async run(input_frame: Frame) → list[Frame]
```

### 4. Processors to Implement

#### a) STTProcessor
- **Provider:** Deepgram (streaming, fast, good accuracy)
- **Input:** Audio frame (raw PCM or wav bytes)
- **Output:** Text frame (transcription)
- **Streaming:** Yes — emit partial transcriptions, final on silence detection

#### b) MemoryRetrieverProcessor
- **Input:** Text frame (user's message)
- **Action:** Search memory store for relevant past context
- **Output:** Memory frame (relevant memories as context string) + pass-through text frame
- **v0.01 scope:** Simple vector similarity search. Top 5 results. No knowledge graph.

#### c) LLMProcessor
- **Provider:** OpenAI GPT-4o (supports text + vision)
- **Input:** Text frame + Memory frame + optional Vision frame
- **Output:** Text frame (LLM response)
- **Streaming:** Yes — stream tokens as they arrive
- **System prompt:** Conversational, warm, references memories naturally
- **After response:** Store the conversation turn in memory

#### d) TTSProcessor
- **Provider:** OpenAI TTS (cost-effective, good quality, simple API)
- **Input:** Text frame (LLM response, streamed sentence-by-sentence)
- **Output:** Audio frame (speech audio bytes)
- **Optimization:** Start synthesizing first sentence while LLM generates second

#### e) VisionProcessor
- **Input:** Vision frame (base64 image from client)
- **Output:** Attaches image to the LLM context (not a separate processor call — feeds into LLMProcessor)
- **v0.01 scope:** Pass image directly to GPT-4o multimodal. No preprocessing.

### 5. Memory Store

- **Database:** SQLite via aiosqlite
- **Embeddings:** OpenAI text-embedding-3-small
- **Schema:**
  - `conversations` table: id, user_message, assistant_message, embedding (blob), timestamp
- **Operations:**
  - `add(user_msg, assistant_msg)` — embed and store
  - `search(query, limit=5)` — cosine similarity search
- **v0.01 scope:** Flat table, brute-force cosine similarity. No HNSW, no chunking strategy. Works fine for <10K conversations.

### 6. WebSocket Server

- **Framework:** FastAPI with WebSocket endpoint
- **Protocol:**
  - Client sends: `{"type": "audio", "data": "<base64 pcm>"}` or `{"type": "image", "data": "<base64 jpg>"}`
  - Server sends: `{"type": "audio", "data": "<base64 pcm>"}` or `{"type": "text", "data": "transcription/response"}`
  - Control messages: `{"type": "control", "action": "start|stop|interrupt"}`
- **Single user for v0.01.** No auth, no sessions, no multi-tenancy.

### 7. Web Client

- **Single HTML file** with inline JS
- **UI:** Centered mic button, waveform visualization (simple canvas), text display for transcription
- **Captures:** Mic audio via MediaRecorder API, camera via getUserMedia
- **Sends:** Audio chunks over WebSocket, images on demand
- **Plays:** Audio responses via Web Audio API
- **No framework.** Vanilla HTML/JS/CSS.

## File Structure

```
app/
├── pyproject.toml          # UV project config
├── README.md
├── src/
│   └── aether/
│       ├── __init__.py
│       ├── main.py             # FastAPI app + WebSocket server
│       ├── core/
│       │   ├── __init__.py
│       │   ├── frames.py       # Frame dataclass
│       │   ├── processor.py    # Processor base class
│       │   └── pipeline.py     # Pipeline runner
│       ├── processors/
│       │   ├── __init__.py
│       │   ├── stt.py          # Deepgram STT
│       │   ├── llm.py          # OpenAI LLM
│       │   ├── tts.py          # OpenAI TTS
│       │   ├── memory.py       # Memory retriever processor
│       │   └── vision.py       # Vision frame handler
│       ├── memory/
│       │   ├── __init__.py
│       │   └── store.py        # SQLite + embeddings memory store
│       └── static/
│           └── index.html      # Web client (single file)
└── .env                        # API keys (OPENAI_API_KEY, DEEPGRAM_API_KEY)
```

## Dependencies (minimal)

```
fastapi
uvicorn[standard]
websockets
openai
deepgram-sdk
aiosqlite
numpy              # for cosine similarity
python-dotenv
```

## What We're NOT Building in v0.01

- No model routing / fallback chains
- No knowledge graph
- No emotion detection
- No WebRTC (WebSocket is fine for prototype)
- No mobile app
- No multi-user / auth
- No provider abstraction layer (hardcoded to OpenAI + Deepgram)
- No parallel streaming optimization (sequential is fine for v0.01)
- No proactive memory surfacing
- No plugin system
- No tests (ship first, test when interfaces stabilize at v0.1)

## Success Criteria

1. User speaks into mic on web page
2. System transcribes speech in <500ms
3. System retrieves relevant memories (if any exist)
4. System generates conversational response via LLM
5. System plays back audio response
6. Total voice-to-voice latency <3 seconds (acceptable for v0.01, optimize later)
7. System remembers previous conversations across page refreshes
8. User can share a camera image and the system responds about what it sees
