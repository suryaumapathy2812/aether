# Local Development (No Docker)

Use this when you want to run app services directly on your machine and only use an external Postgres DB.
It uses the same `DATABASE_URL` variable used by containerized deployment.

## Prerequisites

- `uv`
- `bun`
- `caddy`
- Optional: `cloudflared` (for a shareable tunnel URL)

## Environment

In `.env`, set:

- `DATABASE_URL` (used by both orchestrator and dashboard)

Recommended local values:

- `BETTER_AUTH_URL=https://localhost:3000`
- `BETTER_AUTH_TRUSTED_ORIGINS=https://localhost:3000,http://localhost:3000`
- `CORS_ORIGINS=https://localhost:3000,http://localhost:3000`

## Start / Stop

```bash
./caddy.sh start
./caddy.sh stop
```

Also available:

```bash
./caddy.sh status
./caddy.sh restart
```

## What the script starts

- Agent app (hot reload): `127.0.0.1:8000`
- Orchestrator (hot reload, local-agent mode): `127.0.0.1:9000`
- Dashboard (hot reload): `127.0.0.1:3100`
- Caddy reverse proxy (single origin): `https://localhost:3000`
- Optional cloudflared quick tunnel to `http://localhost:3080`

Logs are written to `.run/local-dev/logs/`.
