# Aether — Deployment Guide

## Architecture

Aether uses a **hybrid deployment** model:

| Component | Runs as | Managed by |
|---|---|---|
| **Dashboard** (Next.js) | Native process on host | PM2 |
| **Orchestrator** (Go) | Native process on host | PM2 |
| **Postgres** | Docker container | docker-compose |
| **MinIO** (S3 storage) | Docker container | docker-compose |
| **Dozzle** (log viewer) | Docker container | docker-compose |
| **Agent** (per-user) | Docker container | Orchestrator (Docker SDK) |
| **Caddy** (reverse proxy) | Docker container | docker run (host networking) |

```
Internet
  │
  ▼
Caddy (:80/:443, host network)
  ├── /api/auth/*    → Dashboard (:3000)
  ├── /api/go/*      → Dashboard (:3000)  → proxies to Orchestrator
  ├── /api/*         → Orchestrator (:4000)
  └── /*             → Dashboard (:3000)
                          │
                     Orchestrator (:4000)
                          │
                    ┌─────┴─────┐
                    ▼           ▼
              Agent (user-1)  Agent (user-2) ...
                    │           │
              ┌─────┴─────┬─────┘
              ▼           ▼
          Postgres    MinIO (aether-minio:9000)
        (localhost:5432)
```

Agent containers join the `aether_internal` Docker network, giving them access to
Postgres and MinIO by container name. The orchestrator provisions and destroys them
dynamically based on user activity.

## Files

```
docker/
  .env.example                 # Environment template — copy to .env
  Caddyfile                    # Caddy reverse proxy rules
  docker-compose.services.yml  # Postgres, MinIO, Dozzle
  ecosystem.config.js          # PM2 config (Dashboard + Orchestrator)
  setup.sh                     # Interactive VPS bootstrap script
  README.md                    # This file
```

## Prerequisites

- Linux VPS (Ubuntu/Debian recommended)
- Docker + Docker Compose v2
- Node.js 20+
- Go 1.22+
- PM2 (`npm install -g pm2`)
- Domain pointing to the VPS IP

## Quick Start (setup.sh)

The interactive setup script handles everything:

```bash
cd /opt/aether
chmod +x docker/setup.sh
./docker/setup.sh
```

It will:
1. Prompt for domain, secrets, API keys, OAuth credentials
2. Generate `/opt/aether/.env`
3. Install prerequisites (Node.js, Go, Docker, PM2)
4. Start Docker services (Postgres, MinIO, Dozzle)
5. Build the Dashboard (`npm install && npm run build`)
6. Run database migrations (`npx prisma db push`)
7. Build the Orchestrator (`go build`)
8. Pre-pull the agent Docker image
9. Start PM2 processes (Dashboard + Orchestrator)
10. Launch Caddy with host networking for HTTPS

## Manual Deployment

### 1. Configure environment

```bash
cp docker/.env.example docker/.env
# Edit docker/.env — fill in all required values
```

### 2. Start Docker services

```bash
cd /opt/aether/docker
docker compose -f docker-compose.services.yml up -d
```

This starts:
- **Postgres** on port `5432` (container: `aether-postgres`)
- **MinIO** on ports `9000`/`9001` (container: `aether-minio`)
- **Dozzle** on port `8080` (container: `aether-dozzle`)

### 3. Build and start Dashboard

```bash
cd /opt/aether/dashboard
npm install
npm run build
npx prisma db push   # creates auth tables, API keys table, etc.
```

### 4. Build Orchestrator

```bash
cd /opt/aether/orchestrator
go build -o aether-orchestrator ./cmd/server
```

### 5. Start PM2

```bash
cd /opt/aether/docker
set -a && source /opt/aether/.env && set +a
pm2 start ecosystem.config.js
pm2 save
pm2 startup   # enable auto-start on reboot
```

### 6. Pre-pull agent image

```bash
docker pull suryaumapathy2812/aether-agent:latest
```

### 7. Start Caddy

```bash
docker run -d \
  --name caddy \
  --network host \
  --restart=unless-stopped \
  -v /opt/aether/docker/Caddyfile:/etc/caddy/Caddyfile:ro \
  -v /var/lib/caddy:/data \
  caddy:latest
```

Caddy runs with `--network host` so it can reach Dashboard (`:3000`) and
Orchestrator (`:4000`) on localhost. It handles TLS automatically via Let's Encrypt.

## Environment Variables Reference

### Required

| Variable | Used by | Description |
|---|---|---|
| `DOMAIN` | Caddyfile, setup.sh | Public domain (e.g., `aether.example.com`) |
| `POSTGRES_PASSWORD` | Compose, PM2 | Postgres password |
| `DATABASE_URL` | Dashboard, Orchestrator | Full Postgres connection string |
| `BETTER_AUTH_SECRET` | Dashboard | Auth token signing key |
| `BETTER_AUTH_URL` | Dashboard | Public dashboard URL for auth callbacks |
| `BETTER_AUTH_TRUSTED_ORIGINS` | Dashboard | Comma-separated allowed origins |
| `AGENT_SECRET` | Orchestrator, Agent | Shared secret for orchestrator-agent auth |

### Agent provisioning

| Variable | Used by | Default | Description |
|---|---|---|---|
| `AGENT_IMAGE` | Orchestrator | `suryaumapathy2812/aether-agent:latest` | Docker image for agent containers |
| `AGENT_NETWORK` | Orchestrator | `aether_internal` | Docker network agents join |
| `AGENT_IDLE_TIMEOUT` | Orchestrator | `1800` | Seconds before idle agent is stopped |
| `AGENT_HEALTH_TIMEOUT` | Orchestrator | `30` | Health check timeout (seconds) |
| `AGENT_PORT` | Orchestrator | `8000` | Port inside agent container |
| `AGENT_ASSETS_ROOT` | Orchestrator | `/var/lib/aether/agents` | Host path for per-user data |

### LLM (forwarded to agents)

| Variable | Used by | Description |
|---|---|---|
| `OPENAI_API_KEY` | Agent | API key for LLM provider |
| `OPENAI_BASE_URL` | Agent | Base URL (e.g., `https://openrouter.ai/api/v1`) |
| `OPENAI_MODEL` | Agent | Model name |
| `AGENT_STATE_KEY` | Agent | Encryption key for sensitive SQLite data |

### S3 / MinIO

| Variable | Used by | Default | Description |
|---|---|---|---|
| `S3_ACCESS_KEY_ID` | Compose, Agent | `minioadmin` | MinIO root user / S3 access key |
| `S3_SECRET_ACCESS_KEY` | Compose, Agent | — | MinIO root password / S3 secret key |
| `S3_ENDPOINT` | Agent | `http://aether-minio:9000` | S3 endpoint (container name on internal network) |
| `S3_BUCKET` | Agent | — | Fixed bucket name (leave empty to use template) |
| `S3_BUCKET_TEMPLATE` | Agent | `core-ai-media-{user}` | Per-user bucket naming pattern |
| `S3_REGION` | Agent | `us-east-1` | S3 region |
| `S3_FORCE_PATH_STYLE` | Agent | `true` | Required for MinIO |

### Optional

| Variable | Used by | Description |
|---|---|---|
| `VAPID_PUBLIC_KEY` | Agent | Web push public key |
| `VAPID_PRIVATE_KEY` | Agent | Web push private key |
| `VAPID_SUBJECT` | Agent | Web push subject (e.g., `mailto:admin@example.com`) |
| `AGENT_UPDATE_REPO` | Agent | GitHub repo for OTA updates |
| `AGENT_UPDATE_TOKEN` | Agent | GitHub token for private release access |
| `GOOGLE_CLIENT_ID` | Agent | Google OAuth client ID (for plugins) |
| `GOOGLE_CLIENT_SECRET` | Agent | Google OAuth client secret |
| `SPOTIFY_CLIENT_ID` | Agent | Spotify client ID (for plugins) |
| `SPOTIFY_CLIENT_SECRET` | Agent | Spotify client secret |
| `DOZZLE_PORT` | Compose | Dozzle web UI port (default: `8080`) |

## Common Operations

### View status

```bash
pm2 status                    # Dashboard + Orchestrator
docker ps                     # Postgres, MinIO, Dozzle, Caddy, agent containers
```

### View logs

```bash
pm2 logs aether-dashboard     # Dashboard logs
pm2 logs aether-orchestrator  # Orchestrator logs
docker logs caddy             # Caddy logs
docker logs aether-postgres   # Postgres logs

# Or use Dozzle web UI at http://<server-ip>:8080
```

### Restart services

```bash
pm2 restart aether-dashboard
pm2 restart aether-orchestrator
docker restart caddy
```

### Update agent image

```bash
docker pull suryaumapathy2812/aether-agent:latest
# Existing agent containers will be recreated on next user request
```

### Update Dashboard / Orchestrator

```bash
cd /opt/aether && git pull

# Dashboard
cd dashboard && npm install && npm run build
pm2 restart aether-dashboard

# Orchestrator
cd ../orchestrator && go build -o aether-orchestrator ./cmd/server
pm2 restart aether-orchestrator
```

### Rebuild everything

```bash
cd /opt/aether
./docker/setup.sh
```

The setup script is idempotent — it loads existing `.env` values as defaults
and only regenerates secrets that are missing.
