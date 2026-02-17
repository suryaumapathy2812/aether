# Aether

A voice-primary AI assistant with a minimal, Light Phone-inspired interface.

```
You speak. Aether listens, thinks, responds. That's it.
```

## Architecture

```
                    ┌──────────────┐
                    │   Dashboard  │  Next.js + Tailwind v4
                    │   :3000      │  430px mobile-first dark UI
                    └──────┬───────┘
                           │
┌──────────┐       ┌──────┴───────┐       ┌──────────────┐
│  iOS App │──────▶│ Orchestrator │──────▶│    Agent     │
│  SwiftUI │       │   :9000      │       │   :8000      │
└──────────┘       │              │       │              │
┌──────────┐       │  Auth, WS    │       │  STT → LLM  │
│ Web/TUI  │──────▶│  proxy,      │       │  → TTS      │
│ Clients  │       │  pairing     │       │  Memory,     │
└──────────┘       └──────┬───────┘       │  Tools,      │
                          │               │  Skills      │
                   ┌──────┴───────┐       └──────────────┘
                   │  PostgreSQL  │
                   │   :5432      │
                   └──────────────┘
```

**Orchestrator** — Auth, agent registry, WebSocket proxy, device pairing, API key management. Clients never talk to the agent directly.

**Agent** — The brain. Streaming voice pipeline (Deepgram STT → OpenAI LLM → TTS), four-tier memory (conversations, facts, actions, sessions), tool execution, skill system, sub-agents.

**Dashboard** — Mobile-first control panel. Login, text chat, memory browser, API key management, device pairing, account settings.

**iOS App** — Pure voice orb interface. Tap to speak, tap to stop. Breathing/pulsing animations. Half-duplex echo cancellation.

## Quick Start

### Prerequisites

- Docker + Docker Compose
- API keys: OpenAI, Deepgram (required), ElevenLabs (optional)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start all services

```bash
docker compose -f docker-compose.dev.yml up --build
```

This starts PostgreSQL, the agent, orchestrator, and dashboard with hot-reload.

### 3. Access

| Service      | URL                          |
|-------------|------------------------------|
| Dashboard   | http://localhost:3000         |
| Orchestrator| http://localhost:9000         |
| Agent       | http://localhost:8000         |
| Agent health| http://localhost:8000/health  |

With OrbStack: `https://dashboard.core-ai.orb.local`, `https://orchestrator.core-ai.orb.local`

## Project Structure

```
aether/
├── app/                    # Agent — Python/FastAPI/UV
│   ├── src/aether/
│   │   ├── main.py         # FastAPI app, WS handler, auto-registration
│   │   ├── core/           # Config, frames, pipeline, logging
│   │   ├── memory/         # Four-tier memory store (SQLite)
│   │   ├── processors/     # STT, LLM, TTS, memory, vision
│   │   ├── providers/      # Deepgram, OpenAI, ElevenLabs, Sarvam
│   │   ├── tools/          # File ops, shell, web search, task runner
│   │   ├── skills/         # Skill loader + marketplace
│   │   └── agents/         # Sub-agent task runner
│   ├── tests/              # pytest suite
│   ├── Dockerfile
│   └── pyproject.toml
│
├── orchestrator/           # Orchestrator — Python/FastAPI/UV
│   ├── src/
│   │   ├── main.py         # Auth, WS proxy, pairing, agent registry
│   │   ├── db.py           # asyncpg + schema bootstrap
│   │   └── auth.py         # JWT + password hashing
│   ├── Dockerfile
│   └── pyproject.toml
│
├── dashboard/              # Dashboard — Next.js 15 + Tailwind v4
│   ├── src/
│   │   ├── app/            # Pages: login, home, chat, memory, services, devices, account
│   │   ├── components/     # PageShell, MinimalInput, MenuList
│   │   └── lib/api.ts      # Orchestrator API client
│   ├── Dockerfile
│   └── Dockerfile.dev
│
├── client/
│   ├── ios/                # iOS — SwiftUI + AVAudioEngine
│   │   ├── Aether/
│   │   │   ├── Views/      # VoiceOrbView (audio pipeline), PairingView
│   │   │   └── Services/   # PairingService, KeychainHelper
│   │   └── project.yml     # XcodeGen spec (regenerate with `xcodegen generate`)
│   ├── web/index.html      # Reference web client (single file)
│   └── tui/                # Terminal UI client (Python/Rich)
│
├── docker-compose.yml      # Production
├── docker-compose.dev.yml  # Development with hot-reload
├── .env.example            # Environment template
└── .gitignore
```

## Voice Protocol

All clients communicate with the agent via WebSocket using JSON text messages:

```
Client → Agent:
  {"type": "stream_start"}              # Start listening
  {"type": "audio_chunk", "data": "..."}  # Base64 PCM16 audio (16kHz)
  {"type": "stream_stop"}               # Stop listening
  {"type": "mute"}                      # Mute mic
  {"type": "unmute"}                    # Unmute mic
  {"type": "text", "data": "hello"}     # Text input

Agent → Client:
  {"type": "transcript", "data": "...", "interim": true/false}
  {"type": "text_chunk", "data": "..."}   # Streaming LLM response
  {"type": "audio_chunk", "data": "..."}  # Base64 TTS audio
  {"type": "stream_start"}               # Agent starts speaking
  {"type": "stream_end"}                 # Agent done speaking
  {"type": "status", "data": "thinking"} # Status updates
```

## iOS Development

```bash
# Prerequisites: Xcode 16+, xcodegen
brew install xcodegen

# Generate Xcode project
cd client/ios
xcodegen generate

# Open in Xcode
open Aether.xcodeproj

# Build for iPhone 17 Pro simulator (Xcode 26 beta)
```

The iOS app uses a pure orb-only UI — no transcription text, no response text. Just the orb with breathing/pulsing animations, a status label, and a mute button.

## Agent Features

### Memory (v0.07)

Four-tier memory system persisted in SQLite:

| Tier | What | Example |
|------|------|---------|
| Conversations | Raw exchanges | "User asked about weather in Tokyo" |
| Facts | Extracted knowledge | "User prefers dark mode" |
| Actions | Tool calls with embeddings | "Wrote file config.yaml" |
| Sessions | Session summaries | "Discussed project setup, used 3 tools" |

Memory is searched via embedding similarity (OpenAI ada-002) and injected into LLM context.

### Tools

Built-in tools available to the LLM:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write/create files |
| `list_directory` | List directory contents |
| `run_command` | Execute shell commands (allowlisted) |
| `web_search` | Search the web via DuckDuckGo |
| `run_task` | Delegate to sub-agent |

### Skills

Skills are loaded from `app/skills/` and matched to user queries by keyword overlap. Each skill injects domain-specific instructions into the LLM system prompt.

### Providers

| Component | Providers |
|-----------|-----------|
| STT | Deepgram (streaming) |
| LLM | OpenAI (GPT-4o) |
| TTS | OpenAI, ElevenLabs, Sarvam |

## Development

### Hot Reload

All services support hot-reload in dev mode:

- **Agent**: uvicorn `--reload` watches `app/src/`
- **Orchestrator**: uvicorn `--reload` watches `orchestrator/src/`
- **Dashboard**: Next.js dev server watches `dashboard/src/`

Source directories are volume-mounted into containers.

### Running Agent Tests

```bash
cd app
uv run pytest tests/ -v
```

### Package Manager

- Python services: **UV** (not pip)
- Dashboard: **npm** (in Docker), **bun** (local)

## Known Issues

This project is in active development. See [GitHub Issues](https://github.com/suryaumapathy2812/aether/issues) for the full list.

Key areas:
- **Security**: Auth tokens in query params, no rate limiting, agent endpoints unauthenticated
- **Architecture**: Single-tenant agent (one user at a time), no multi-user support
- **iOS**: Hardcoded localhost URL, thread safety issues, no audio interruption handling
- **Dashboard**: No server-side route protection, broken save button, no WS reconnection
- **Infra**: Containers run as root, no CI/CD, no TLS termination

## License

Private — not open source.
