# Aether Orchestrator

Go rewrite of the orchestration layer.

## Responsibility

- Dashboard always calls orchestrator.
- Orchestrator authenticates requests from bearer token.
- Orchestrator resolves `user_id -> agent host/port` and proxies to the right agent.

## Implemented route groups

- `GET /health`, `GET /api/health`
- `POST /api/agents/register`
- `POST /api/agents/{id}/heartbeat`
- `POST /api/agents/{id}/assign?user_id=...`
- `GET /api/agents/health`
- Authenticated user-routing proxy:
  - `/v1/*`
  - `/api/memory/*`
  - `/api/plugins*`
  - `/api/push/vapid-key`
  - `/api/push/subscribe`
- Realtime-support endpoints:
  - `GET /api/agent/ready`
  - `POST /api/webrtc/offer`
  - `PATCH /api/webrtc/ice`
  - `WS /api/ws/notifications` (and `WS /api/ws` alias)
- `GET /api/metrics/latency` (proxy if available, otherwise degraded payload)

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

## Run

```bash
cd orchestrator
go run ./cmd/server
```
