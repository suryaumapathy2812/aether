# Core AI

Multi-tenant voice agent platform with a web dashboard, orchestrator control plane, and per-user runtime agents.

## Architecture

- `dashboard/` - Next.js app for auth, settings, plugins, and chat controls.
- `orchestrator/` - FastAPI control plane for auth, plugin config, agent lifecycle, and API routing.
- `app/` - Runtime agent (FastAPI) with voice pipeline, WebRTC transport, tools, skills, and memory.
- `docker/` - All Docker Compose stacks and Docker env templates.

### Runtime topology

1. User interacts with `dashboard`.
2. `dashboard` calls `orchestrator` APIs.
3. `orchestrator` provisions or reuses a per-user `app` agent container.
4. Voice traffic flows directly to agent endpoints (WebRTC / voice pipeline).
5. Shared model assets are mounted from `.cache/models` into orchestrator and agents.

## Repository layout

```text
core-ai/
  app/                        # Agent service (Python/FastAPI)
  orchestrator/               # Orchestrator service (Python/FastAPI)
  dashboard/                  # Dashboard service (Next.js)
  client/                     # Client references (web/ios/tui)
  docker/
    docker-compose.yml
    docker-compose.override.yml
    docker-compose.prod.yml
    .env.example
```

## Environment setup

```bash
cp docker/.env.example docker/.env
```

Fill required values in `docker/.env`:

- `AGENT_SECRET`
- `BETTER_AUTH_SECRET`
- provider keys as needed (`OPENROUTER_API_KEY` for LLM, `OPENAI_API_KEY` for TTS, `DEEPGRAM_API_KEY`, ...)

## Run (development)

```bash
docker compose --env-file docker/.env \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.override.yml \
  up --build
```

## Run (production-style compose)

```bash
docker compose --env-file docker/.env \
  -f docker/docker-compose.prod.yml \
  up -d
```

## Images

- Agent image: `aether-agent:<tag>`
- Orchestrator image: `aether-orchestrator:<tag>`

Local build examples:

```bash
docker build -t aether-agent:local ./app
docker build -t aether-orchestrator:local ./orchestrator
```

## Notes

- Root `.env` files can remain for non-docker workflows, but compose flows now use `docker/.env` via `--env-file`.
- Keep model cache under `.cache/models` for VAD and turn detection assets.
