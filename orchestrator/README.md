# Aether Orchestrator (`orchestrator`)

Control plane for multi-user runtime agents. Handles auth validation, per-user agent lifecycle, plugin config APIs, and proxying to spawned agent containers.

## What this service owns

- Auth/session verification (Better Auth token validation)
- Per-user agent container spawn/reuse/reap
- Device pairing + service key + plugin config endpoints
- WebRTC signaling proxy and chat/memory proxy routes
- Shared model bootstrap and periodic background maintenance

## Current architecture

- API + startup lifecycle: `src/main.py`
- Agent container manager: `src/agent_manager.py`
- Auth adapter: `src/auth.py`
- DB pool + schema bootstrap: `src/db.py`
- Secret encryption helpers: `src/crypto.py`

## Run locally

```bash
uv sync --dev
uv run uvicorn src.main:app --host 0.0.0.0 --port 9000 --reload --reload-dir src
```

## Test

```bash
uv run pytest
```

## Build image

```bash
docker build -t aether-orchestrator:local .
```

## Important env vars

- Core: `DATABASE_URL`, `CORS_ORIGINS`, `BETTER_AUTH_SECRET`, `AGENT_SECRET`
- Agent lifecycle: `AGENT_IMAGE`, `AGENT_NETWORK`, `AGENT_IDLE_TIMEOUT`, `AGENT_DEV_ROOT`
- Shared models: `AGENT_SHARED_MODELS_HOST_PATH`, `AGENT_SHARED_MODELS_ORCH_PATH`, `AGENT_SHARED_MODELS_CONTAINER_PATH`
- Fallback keys passed to agents: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`

## Current constraints

- If `AGENT_SECRET` is unset, internal agent endpoints become weakly protected.
- OAuth state storage is currently in-memory (single-process limitation).
- Dev host-mode agent flow is practical for limited local concurrency.
