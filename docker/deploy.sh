#!/bin/bash
#
# Aether Deployment Script
#
# Builds and deploys all components after code updates.
# Run from project root so .env is loaded automatically.
#
# Usage:
#   ./docker/deploy.sh [options]
#
# Options:
#   --skip-dashboard    Skip dashboard build
#   --skip-orchestrator Skip orchestrator build
#   --skip-agent        Skip agent image build
#   --skip-restart      Skip PM2 restart
#   --skip-cleanup      Skip removing old agent containers
#   --pull              Pull latest code from git first
#   --help              Show this help message
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Determine script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default options
SKIP_DASHBOARD=false
SKIP_ORCHESTRATOR=false
SKIP_AGENT=false
SKIP_RESTART=false
SKIP_CLEANUP=false
DO_PULL=false
CADDY_IMAGE="aether-caddy"

require_env() {
    local var_name="$1"
    local message="$2"
    if [ -z "${!var_name}" ]; then
        echo -e "${RED}Error: ${message}${NC}"
        exit 1
    fi
}

build_caddy_image() {
    local caddy_dir="$1"
    echo "Building custom Caddy image with Cloudflare DNS support..."
    docker build -t "$CADDY_IMAGE" -f "$caddy_dir/Dockerfile" "$caddy_dir"
}

restart_caddy_container() {
    local caddy_dir="$1"
    local caddy_data_dir="$2"
    echo "Restarting Caddy..."
    docker rm -f caddy 2>/dev/null || true
    mkdir -p "$caddy_data_dir"
    docker run -d \
        --name caddy \
        --network host \
        --restart=unless-stopped \
        -e CF_API_TOKEN="$CF_API_TOKEN" \
        -v "$caddy_dir/Caddyfile:/etc/caddy/Caddyfile:ro" \
        -v "$caddy_data_dir:/data" \
        "$CADDY_IMAGE"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-dashboard)
            SKIP_DASHBOARD=true
            shift
            ;;
        --skip-orchestrator)
            SKIP_ORCHESTRATOR=true
            shift
            ;;
        --skip-agent)
            SKIP_AGENT=true
            shift
            ;;
        --skip-restart)
            SKIP_RESTART=true
            shift
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
            shift
            ;;
        --pull)
            DO_PULL=true
            shift
            ;;
        --help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     Aether Deployment Script        ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================
# Load Environment
# ============================================
echo -e "${GREEN}Loading environment...${NC}"

# Check if we're in the project root (should have .env)
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading .env from $PROJECT_ROOT"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
elif [ -f "/opt/aether/.env" ]; then
    echo "Loading .env from /opt/aether"
    set -a
    source "/opt/aether/.env"
    set +a
else
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Expected at: $PROJECT_ROOT/.env or /opt/aether/.env"
    exit 1
fi

require_env "CF_API_TOKEN" "CF_API_TOKEN is required for Caddy wildcard certificates and direct agent subdomains."

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo ""

# ============================================
# Pull Latest Code (Optional)
# ============================================
if [ "$DO_PULL" = true ]; then
    echo -e "${GREEN}Pulling latest code...${NC}"
    git pull
    echo ""
fi

# ============================================
# Build Dashboard (TanStack Start / Vite+)
# ============================================
if [ "$SKIP_DASHBOARD" = false ]; then
    echo -e "${GREEN}Building Dashboard...${NC}"
    cd "$PROJECT_ROOT/client/web/aether"
    
    # Limit memory for build
    export NODE_OPTIONS="--max-old-space-size=2048"
    
    echo "Installing dependencies..."
    pnpm install --frozen-lockfile --ignore-scripts
    
    echo "Generating Prisma client..."
    DATABASE_URL="$DATABASE_URL" pnpm exec prisma generate
    
    echo "Building TanStack Start app..."
    pnpm exec vp build
    
    echo -e "${GREEN}Dashboard build complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping dashboard build${NC}"
    echo ""
fi

# ============================================
# Build Orchestrator
# ============================================
if [ "$SKIP_ORCHESTRATOR" = false ]; then
    echo -e "${GREEN}Building Orchestrator...${NC}"
    cd "$PROJECT_ROOT/orchestrator"
    
    echo "Compiling Go binary..."
    go build -o aether-orchestrator ./cmd/server
    
    echo -e "${GREEN}Orchestrator build complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping orchestrator build${NC}"
    echo ""
fi

# ============================================
# Build Agent Docker Image
# ============================================
if [ "$SKIP_AGENT" = false ]; then
    echo -e "${GREEN}Building Agent Docker Image...${NC}"
    cd "$PROJECT_ROOT/agent"
    
    # Get version info for build args
    VERSION=$(git describe --tags --always 2>/dev/null || echo "dev")
    COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Determine image name
    AGENT_IMAGE="${AGENT_IMAGE:-suryaumapathy2812/aether-agent:latest}"
    
    echo "Building image: $AGENT_IMAGE"
    echo "  Version: $VERSION"
    echo "  Commit: $COMMIT"
    echo "  Build Time: $BUILD_TIME"
    
    docker build \
        --build-arg VERSION="$VERSION" \
        --build-arg COMMIT="$COMMIT" \
        --build-arg BUILD_TIME="$BUILD_TIME" \
        -t "$AGENT_IMAGE" \
        .
    
    echo -e "${GREEN}Agent image build complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping agent image build${NC}"
    echo ""
fi

# ============================================
# Stop and Remove Old Agent Containers
# ============================================
if [ "$SKIP_CLEANUP" = false ]; then
    echo -e "${GREEN}Cleaning up old agent containers...${NC}"
    
    # Agent containers are named like "aether-agent-<user-id>"
    AGENT_CONTAINERS=$(docker ps -a --filter "name=aether-agent-" --format "{{.Names}}" 2>/dev/null || true)
    
    if [ -n "$AGENT_CONTAINERS" ]; then
        echo "Found agent containers:"
        echo "$AGENT_CONTAINERS" | while read -r container; do
            echo "  - $container"
        done
        
        echo ""
        echo "Stopping containers..."
        echo "$AGENT_CONTAINERS" | xargs -r docker stop 2>/dev/null || true
        
        echo "Removing containers..."
        echo "$AGENT_CONTAINERS" | xargs -r docker rm 2>/dev/null || true
        
        echo -e "${GREEN}Agent containers cleaned up!${NC}"
    else
        echo "No agent containers found to clean up."
    fi
    echo ""
else
    echo -e "${YELLOW}Skipping agent container cleanup${NC}"
    echo ""
fi

# ============================================
# Restart PM2 Services
# ============================================
if [ "$SKIP_RESTART" = false ]; then
    echo -e "${GREEN}Restarting PM2 services...${NC}"
    
    # Check if PM2 is running these apps
    if pm2 describe aether-dashboard > /dev/null 2>&1; then
        echo "Restarting aether-dashboard..."
        pm2 restart aether-dashboard --update-env
    else
        echo -e "${YELLOW}aether-dashboard not found in PM2, starting fresh...${NC}"
        cd "$PROJECT_ROOT/docker"
        pm2 start ecosystem.config.js --only aether-dashboard
    fi

    if pm2 describe aether-orchestrator > /dev/null 2>&1; then
        echo "Restarting aether-orchestrator..."
        pm2 restart aether-orchestrator --update-env
    else
        echo -e "${YELLOW}aether-orchestrator not found in PM2, starting fresh...${NC}"
        cd "$PROJECT_ROOT/docker"
        pm2 start ecosystem.config.js --only aether-orchestrator
    fi
    
    # Save PM2 state
    pm2 save
    
    echo -e "${GREEN}PM2 services restarted!${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping PM2 restart${NC}"
    echo ""
fi

# ============================================
# Rebuild and Restart Caddy
# ============================================
echo -e "${GREEN}Refreshing Caddy...${NC}"
build_caddy_image "$PROJECT_ROOT/caddy"
restart_caddy_container "$PROJECT_ROOT/caddy" "/var/lib/caddy"
echo -e "${GREEN}Caddy refresh complete!${NC}"
echo ""

# ============================================
# Summary
# ============================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}     Deployment Complete!            ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Status:"
pm2 status
echo ""
echo -e "Agent Image:"
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | grep -E "(REPOSITORY|aether-agent)" || true
echo ""
echo -e "Commands:"
echo -e "  - View logs: pm2 logs"
echo -e "  - Check status: pm2 status"
echo -e "  - View agent containers: docker ps --filter 'name=aether-agent-'"
echo ""
