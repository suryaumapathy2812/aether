#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT_DIR/.run/local-dev"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

# ---- Defaults for local dev ----
export DOMAIN="${DOMAIN:-localhost:3000}"
export BETTER_AUTH_URL="${BETTER_AUTH_URL:-https://localhost:3000}"
export BETTER_AUTH_TRUSTED_ORIGINS="${BETTER_AUTH_TRUSTED_ORIGINS:-https://localhost:3000,http://localhost:3000}"
export CORS_ORIGINS="${CORS_ORIGINS:-https://localhost:3000,http://localhost:3000}"

APP_DB_URL="${DATABASE_URL:-}"

if [[ -z "$APP_DB_URL" ]]; then
  echo "Missing DATABASE_URL in .env"
  exit 1
fi

start_proc() {
  local name="$1"
  local workdir="$2"
  local cmd="$3"
  local pidfile="$PID_DIR/$name.pid"
  local logfile="$LOG_DIR/$name.log"

  if [[ -f "$pidfile" ]]; then
    local existing_pid
    existing_pid="$(cat "$pidfile")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      echo "$name already running (pid $existing_pid)"
      return
    fi
    rm -f "$pidfile"
  fi

  (
    cd "$workdir"
    nohup bash -lc "$cmd" >"$logfile" 2>&1 &
    echo $! >"$pidfile"
  )

  local pid
  pid="$(cat "$pidfile")"
  echo "Started $name (pid $pid)"
}

stop_proc() {
  local name="$1"
  local pidfile="$PID_DIR/$name.pid"
  if [[ ! -f "$pidfile" ]]; then
    return
  fi

  local pid
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 0.5
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    echo "Stopped $name"
  fi
  rm -f "$pidfile"
}

show_cloudflared_url() {
  local logfile="$LOG_DIR/cloudflared.log"
  if [[ -f "$logfile" ]]; then
    local url
    url="$(rg -o 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' "$logfile" -n --no-heading | tail -n 1 | cut -d: -f2-)"
    if [[ -n "$url" ]]; then
      echo "Cloudflare URL: $url"
    fi
  fi
}

start_all() {
  command -v caddy >/dev/null 2>&1 || { echo "caddy is required"; exit 1; }
  command -v uv >/dev/null 2>&1 || { echo "uv is required"; exit 1; }
  command -v bun >/dev/null 2>&1 || { echo "bun is required"; exit 1; }

  start_proc "agent" "$ROOT_DIR/app" \
    "ORCHESTRATOR_URL=http://127.0.0.1:9000 AETHER_AGENT_HOST=127.0.0.1 uv run uvicorn aether.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src"

  start_proc "orchestrator" "$ROOT_DIR/orchestrator" \
    "DATABASE_URL='$APP_DB_URL' AETHER_LOCAL_AGENT_URL=http://127.0.0.1:8000 uv run uvicorn src.main:app --host 0.0.0.0 --port 9000 --reload --reload-dir src"

  start_proc "dashboard" "$ROOT_DIR/dashboard" \
    "DATABASE_URL='$APP_DB_URL' ORCHESTRATOR_URL=http://127.0.0.1:9000 BETTER_AUTH_URL='$BETTER_AUTH_URL' BETTER_AUTH_TRUSTED_ORIGINS='$BETTER_AUTH_TRUSTED_ORIGINS' NEXT_PUBLIC_ORCHESTRATOR_URL= bun run dev -- --port 3100"

  start_proc "caddy" "$ROOT_DIR" \
    "caddy run --config '$ROOT_DIR/Caddyfile.local' --adapter caddyfile"

  if command -v cloudflared >/dev/null 2>&1; then
    start_proc "cloudflared" "$ROOT_DIR" \
      "cloudflared tunnel --no-autoupdate --url http://localhost:3080"
    sleep 3
    show_cloudflared_url
  else
    echo "cloudflared not found, skipping tunnel"
  fi

  echo ""
  echo "Dashboard: https://localhost:3000"
  echo "WebSocket: wss://localhost:3000/api/ws"
  echo "Logs: $LOG_DIR"
}

stop_all() {
  stop_proc "cloudflared"
  stop_proc "caddy"
  stop_proc "dashboard"
  stop_proc "orchestrator"
  stop_proc "agent"
}

status_all() {
  for name in agent orchestrator dashboard caddy cloudflared; do
    local pidfile="$PID_DIR/$name.pid"
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      echo "$name: running (pid $(cat "$pidfile"))"
    else
      echo "$name: stopped"
    fi
  done
  show_cloudflared_url
}

case "${1:-start}" in
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  restart)
    stop_all
    start_all
    ;;
  status)
    status_all
    ;;
  *)
    echo "Usage: ./caddy.sh {start|stop|restart|status}"
    exit 1
    ;;
esac
