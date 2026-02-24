# Aether Agent (`app`)

Runtime AI agent service for text + voice interactions. It exposes OpenAI-compatible chat APIs, WebRTC voice transport, plugin hooks, and memory-backed context retrieval.

## What this service owns

- FastAPI runtime + transport endpoints (`/chat`, `/v1/chat/completions`, `/webrtc/*`)
- Voice pipeline (STT -> turn detection/VAD -> LLM -> TTS)
- Tool/skill/plugin loading and execution
- SQLite memory (`conversations`, `facts`, `actions`, `sessions`)
- Agent registration + heartbeat integration with orchestrator

## Current architecture

- Composition root: `src/aether/main.py`
- Core facade: `src/aether/agent.py` (`AgentCore`)
- Scheduler + job lanes: `src/aether/kernel/scheduler.py`
- LLM flow: `src/aether/services/reply_service.py` + `src/aether/llm/*`
- Voice/WebRTC: `src/aether/voice/session.py`, `src/aether/voice/webrtc.py`
- Memory store: `src/aether/memory/store.py`

### P/E execution split

- P-worker ingress (always-on, multimodal): `/chat`, `/v1/chat/completions`, `/webrtc/*`
- P-worker model: Gemini realtime session (requires `GEMINI_API_KEY`)
- E-worker execution: delegated/advanced agentic workflows (`delegate_to_agent`, background tasks)
- User ingress does not silently fall back to scheduler/OpenRouter when P-worker realtime is unavailable

## Run locally

```bash
uv sync --extra webrtc --dev
uv run uvicorn aether.main:app --host 0.0.0.0 --port 8000
```

## Test

```bash
uv run python -m pytest
```

## Build image

```bash
docker build -t aether-agent:local .
```

## Important env vars

- Runtime: `AETHER_HOST`, `AETHER_PORT`, `AETHER_DB_PATH`, `AETHER_WORKING_DIR`
- LLM/TTS/STT: `AETHER_LLM_MODEL`, `AETHER_TTS_PROVIDER`, `AETHER_STT_PROVIDER`
- Keys: `GEMINI_API_KEY` (required for P-worker voice/text realtime ingress), `OPENROUTER_API_KEY` (agentic/background LLM paths), `OPENAI_API_KEY` (TTS), `DEEPGRAM_API_KEY` (STT)
- Voice tuning: `AETHER_VAD_MODE`, `AETHER_TURN_DETECTION_MODE`, `AETHER_WEBRTC_SESSION_TTL`

## Current constraints

- WebRTC routes require `aiortc`; without it `/webrtc/*` is unavailable.
- Scheduler uses queue limits and can reject/shed jobs under load.
- Service is currently optimized for one active user/session context per agent runtime.
