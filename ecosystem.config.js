// ecosystem.local.js - PM2 configuration for local development
// Usage: pm2 start ecosystem.local.js
//
// Prerequisites:
//   1. Docker services running: cd docker && docker compose -f docker-compose.services.yml up -d
//   2. cloudflared running:     cloudflared tunnel --url localhost:80
//   3. Set CHANNELS_WEBHOOK_URL in agent/.env to the cloudflared URL

const ROOT = "/Users/suryaumapathy/Developers/Github/suryaumapathy/core-ai";

module.exports = {
  apps: [
    // ─────────────────────────────────────────────
    // Dashboard (TanStack Start / Vite+) — port 3000
    // ─────────────────────────────────────────────
    {
      name: "aether-dashboard",
      script: "bash",
      args: `-l -c 'source ~/.vite-plus/env && vp dev --port 3000'`,
      cwd: `${ROOT}/client/web/aether`,
      interpreter: "none",
      autorestart: true,
      watch: false,
      env: {
        NODE_ENV: "development",
      },
    },

    // ─────────────────────────────────────────────
    // Orchestrator (Go) — port 4000
    // Reads config from orchestrator/.env
    // ─────────────────────────────────────────────
    {
      name: "aether-orchestrator",
      script: "bash",
      args: "-c 'go run ./cmd/server'",
      cwd: `${ROOT}/orchestrator`,
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 2000,
    },

    // ─────────────────────────────────────────────
    // Agent (Go) — port 8000
    // Reads config from agent/.env
    // ─────────────────────────────────────────────
    {
      name: "aether-agent",
      script: "bash",
      args: "-c 'go run ./cmd/server'",
      cwd: `${ROOT}/agent`,
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 2000,
    },

    // ─────────────────────────────────────────────
    // Caddy (reverse proxy) — port 80
    // ─────────────────────────────────────────────
    {
      name: "aether-caddy",
      script: "caddy",
      args: `run --config ${ROOT}/caddy/Caddyfile.local`,
      cwd: `${ROOT}/caddy`,
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_restarts: 5,
      restart_delay: 1000,
    },
  ],
};
