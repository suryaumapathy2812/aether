# Aether Orchestrator

Go rewrite of the orchestration layer.

## Responsibility

- Dashboard always calls orchestrator.
- Orchestrator authenticates requests from bearer token.
- Orchestrator resolves `user_id -> agent host/port` and proxies to the right agent.

## Implemented route groups

- `GET /health`, `GET /api/health`, `GET /go/v1/health`
- `POST /go/v1/agents/register`
- `POST /go/v1/agents/{id}/heartbeat`
- `POST /go/v1/agents/{id}/assign?user_id=...`
- `GET /go/v1/agents/health`
- Admin endpoints (require `AGENT_SECRET` bearer when configured):
  - `GET /go/v1/agents/version`
  - `POST /go/v1/agents/reload`
  - `POST /go/v1/agents/upgrade`
- Authenticated user-routing proxy:
  - `/agent/v1/*`
- Realtime-support endpoints:
  - `GET /go/v1/agent/ready`
  - `GET /go/v1/agent/subdomain`
  - `WS /agent/v1/ws/notifications` (with legacy `/api/ws/notifications` and `/api/ws` aliases)
  - `WS /agent/v1/ws/conversation` (with legacy `/api/ws/conversation` alias)
- `GET /go/v1/metrics/latency` (proxy if available, otherwise degraded payload)

## Behavior guarantees

- Token-derived `user_id` is enforced for routed traffic.
- For known user-scoped calls, `user_id` query/body fields are rewritten to the
  authenticated user.
- HTTP proxy preserves status, content-type, and streaming response bodies.
- Notification websocket is proxied through orchestrator to the assigned agent.

## Configuration

- `PORT` (default `4000`)
- `DATABASE_URL` (required)
- `AGENT_SECRET` (optional; protects agent registration endpoints)
- `AETHER_LOCAL_AGENT_URL` (optional; single-agent dev mode override)
- `AGENT_PROXY_TIMEOUT_SECONDS` (default `120`)
- `AETHER_DEFAULT_AGENT_ID` (optional fallback assignment)
- `AETHER_AUTO_ASSIGN_FIRST_AGENT` (`true` to assign first running agent when user has no assignment)
- `AGENT_IMAGE` (default `suryaumapathy2812/aether-agent:latest`)
- `AGENT_NETWORK` (optional; auto-detected when running in Docker)
- `AGENT_IDLE_TIMEOUT` (seconds, default `1800`)
- `AGENT_HEALTH_TIMEOUT` (seconds, default `30`)
- `AGENT_PORT` (default `8000`)
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` (forwarded into provisioned agents)
- `AGENT_STATE_KEY` (optional; forwarded for agent SQLite encryption)
- `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`, `VAPID_SUBJECT` (optional; forwarded for web push)
- `S3_BUCKET`, `S3_BUCKET_TEMPLATE`, `S3_REGION`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_ENDPOINT`, `S3_PUBLIC_BASE_URL`, `S3_FORCE_PATH_STYLE`, `S3_PUT_URL_TTL_SECONDS`, `S3_GET_URL_TTL_SECONDS` (forwarded into provisioned agents)
- `AGENT_UPDATE_REPO` (default `suryaumapathy2812/aether`)
- `AGENT_UPDATE_TOKEN` (optional GitHub token for private releases)

## Per-user persistence

- Each user gets one dedicated agent container.
- Each agent mounts a dedicated persistent Docker volume at `/app/assets`.
- The user's SQLite DB (`state.db`) lives in that volume and survives container restarts/upgrades.

## Run

```bash
cd orchestrator
go run ./cmd/server
```
