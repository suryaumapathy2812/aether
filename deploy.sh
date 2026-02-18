#!/usr/bin/env bash
#
# Aether — Deploy Script
# One command to bring up the entire stack.
#
# Usage:
#   ./deploy.sh          # Dev mode (default)
#   ./deploy.sh prod     # Production mode (multi-user, auto-HTTPS)
#   ./deploy.sh down     # Stop everything
#   ./deploy.sh logs     # Tail logs
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

info()  { echo -e "${CYAN}▸${NC} $1"; }
ok()    { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "${RED}✗${NC} $1"; exit 1; }

MODE="${1:-dev}"

# ── Down / Logs shortcuts ──────────────────────────────────

if [ "$MODE" = "down" ]; then
    info "Stopping all services..."
    docker compose -f docker-compose.dev.yml down 2>/dev/null || true
    docker compose down 2>/dev/null || true
    ok "All services stopped"
    exit 0
fi

if [ "$MODE" = "logs" ]; then
    COMPOSE_FILE="docker-compose.dev.yml"
    [ -f docker-compose.override.yml ] && COMPOSE_FILE="docker-compose.yml"
    exec docker compose -f "$COMPOSE_FILE" logs -f --tail=100
fi

# ── Prerequisites ──────────────────────────────────────────

info "Checking prerequisites..."

command -v docker >/dev/null 2>&1 || fail "Docker not found. Install: https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 || fail "Docker daemon not running. Start Docker Desktop or dockerd."

# Check docker compose (v2 plugin)
if docker compose version >/dev/null 2>&1; then
    ok "Docker Compose $(docker compose version --short)"
else
    fail "Docker Compose not found. Install: https://docs.docker.com/compose/install/"
fi

# ── Environment ────────────────────────────────────────────

if [ ! -f .env ]; then
    info "No .env found — creating from .env.example"
    cp .env.example .env

    # Check for required keys
    if grep -q "^OPENAI_API_KEY=sk-\.\.\." .env; then
        echo ""
        warn "Required API keys not set in .env:"
        echo -e "  ${DIM}OPENAI_API_KEY${NC}  — https://platform.openai.com/api-keys"
        echo -e "  ${DIM}DEEPGRAM_API_KEY${NC} — https://console.deepgram.com"
        echo ""
        echo -e "  Edit ${CYAN}.env${NC} and fill in your keys, then re-run this script."
        exit 1
    fi
else
    ok ".env exists"
fi

# Source .env for variable access
set -a
source .env
set +a

# ── Build & Deploy ─────────────────────────────────────────

if [ "$MODE" = "prod" ]; then
    COMPOSE_FILE="docker-compose.yml"
    info "Production mode"

    # Build agent image (needed for multi-user dynamic provisioning)
    info "Building agent image..."
    docker build -t "${AGENT_IMAGE:-core-ai-agent:latest}" ./app
    ok "Agent image built"

    info "Starting services..."
    docker compose -f "$COMPOSE_FILE" up --build -d
else
    COMPOSE_FILE="docker-compose.dev.yml"
    info "Development mode"

    info "Starting services..."
    docker compose -f "$COMPOSE_FILE" up --build -d
fi

# ── Health Checks ──────────────────────────────────────────

info "Waiting for services to be ready..."

# Wait for Postgres
RETRIES=30
until docker compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U aether >/dev/null 2>&1; do
    RETRIES=$((RETRIES - 1))
    [ $RETRIES -le 0 ] && fail "Postgres failed to start"
    sleep 1
done
ok "Postgres ready"

# Wait for orchestrator
RETRIES=30
until curl -sf http://localhost:3000/health >/dev/null 2>&1; do
    RETRIES=$((RETRIES - 1))
    [ $RETRIES -le 0 ] && { warn "Orchestrator health check timed out (may still be starting)"; break; }
    sleep 2
done
[ $RETRIES -gt 0 ] && ok "Orchestrator ready"

# Wait for dashboard (via Caddy)
RETRIES=15
until curl -sf http://localhost:3000 >/dev/null 2>&1; do
    RETRIES=$((RETRIES - 1))
    [ $RETRIES -le 0 ] && { warn "Dashboard not responding yet (may still be compiling)"; break; }
    sleep 2
done
[ $RETRIES -gt 0 ] && ok "Dashboard ready"

# ── Summary ────────────────────────────────────────────────

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Aether is running${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$MODE" = "prod" ]; then
    DOMAIN="${DOMAIN:-localhost}"
    echo -e "  Dashboard:   ${CYAN}https://${DOMAIN}${NC}"
    echo -e "  WebSocket:   ${CYAN}wss://${DOMAIN}/ws${NC}"
else
    echo -e "  Dashboard:   ${CYAN}http://localhost:3000${NC}"
    echo -e "  WebSocket:   ${CYAN}ws://localhost:3000/ws${NC}"
    echo -e "  Postgres:    ${DIM}localhost:5432${NC}"
fi

echo ""
echo -e "  ${DIM}Logs:  ./deploy.sh logs${NC}"
echo -e "  ${DIM}Stop:  ./deploy.sh down${NC}"
echo ""
