#!/bin/bash
#
# Aether Setup Script
#
# Interactive setup for Aether on a new VPS
# 
# Usage:
#   chmod +x setup.sh && ./setup.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values (will be overridden by .env.example or .env if present)
DEFAULT_DOMAIN="aether.suryaumapathy.in"
DEFAULT_OPENAI_BASE_URL="https://openrouter.ai/api/v1"
DEFAULT_OPENAI_MODEL="google/gemini-2.5-flash"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     Aether Setup Script            ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root is not recommended.${NC}"
    echo ""
fi

# Determine Aether directory
AETHER_DIR="/opt/aether"
ENV_FILE="$AETHER_DIR/.env"
ENV_EXAMPLE="$AETHER_DIR/docker/.env.example"

# Load .env.example first (provides base defaults)
if [ -f "$ENV_EXAMPLE" ]; then
    echo -e "${GREEN}Loading defaults from $ENV_EXAMPLE${NC}"
    set -a
    source "$ENV_EXAMPLE"
    set +a
fi

# Load existing .env if present (overrides .env.example)
if [ -f "$ENV_FILE" ]; then
    echo -e "${GREEN}Found existing .env file at $ENV_FILE${NC}"
    echo "Loading existing configuration..."
    set -a
    source "$ENV_FILE"
    set +a
fi

# Helper function to prompt with default value
prompt_with_default() {
    local prompt="$1"
    local var_name="$2"
    local default_value="$3"
    local is_required="$4"  # "yes" or "no"
    
    local current_value="${!var_name}"
    local display_default="${current_value:-$default_value}"
    
    if [ "$is_required" = "yes" ]; then
        echo -e "${prompt}"
        echo -e "  (required, press Enter to keep: ${display_default})"
    else
        echo -e "${prompt}"
        echo -e "  (press Enter to keep: ${display_default} or 'skip' to leave empty)"
    fi
    
    read -r input
    
    if [ -z "$input" ]; then
        # User pressed Enter - keep current value
        if [ -z "$current_value" ]; then
            eval "$var_name=$default_value"
        else
            eval "$var_name=$current_value"
        fi
    elif [ "$input" = "skip" ]; then
        eval "$var_name="
    else
        eval "$var_name=$input"
    fi
    
    echo ""
}

# Helper for yes/no prompts
prompt_yes_no() {
    local prompt="$1"
    local default="$2"
    
    local current_value="${!default}"
    local display_default=$([ "$current_value" = "yes" ] && echo "Y/n" || echo "n/Y")
    
    echo -e "${prompt}"
    echo -e "  ($display_default)"
    read -r input
    
    if [ -z "$input" ]; then
        eval "$default=$current_value"
    elif [ "$input" = "y" ] || [ "$input" = "Y" ] || [ "$input" = "yes" ]; then
        eval "$default=yes"
    else
        eval "$default=no"
    fi
    echo ""
}

# ============================================
# Step 1: Docker Hub Login (Optional)
# ============================================
echo -e "${GREEN}Step 1: Docker Hub Login (Optional)${NC}"
echo "Docker Hub has rate limits. Login to avoid issues pulling images."
echo ""
prompt_with_default "Docker Hub Username (press Enter to skip):" "DOCKERHUB_USERNAME" "" "no"

if [ -n "$DOCKERHUB_USERNAME" ]; then
    prompt_with_default "Docker Hub Access Token (https://hub.docker.com/settings/security):" "DOCKERHUB_TOKEN" "" "no"
fi

# ============================================
# Step 2: Domain Configuration
# ============================================
echo -e "${GREEN}Step 2: Domain Configuration${NC}"
prompt_with_default "Enter your domain (e.g., aether.suryaumapathy.in):" "DOMAIN" "$DEFAULT_DOMAIN" "yes"

# ============================================
# Step 3: Security Configuration
# ============================================
echo -e "${GREEN}Step 3: Security Configuration${NC}"

generate_password() {
    openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32
}

# Check if we already have passwords
if [ -z "$POSTGRES_PASSWORD" ]; then
    POSTGRES_PASSWORD=$(generate_password)
    echo "Generated new POSTGRES_PASSWORD"
fi

if [ -z "$BETTER_AUTH_SECRET" ]; then
    BETTER_AUTH_SECRET=$(generate_password)
    echo "Generated new BETTER_AUTH_SECRET"
fi

if [ -z "$AGENT_SECRET" ]; then
    AGENT_SECRET=$(generate_password)
    echo "Generated new AGENT_SECRET"
fi

if [ -z "$S3_SECRET" ]; then
    S3_SECRET=$(generate_password)
    echo "Generated new S3_SECRET_ACCESS_KEY"
fi

echo -e "${YELLOW}Please save these passwords:${NC}"
echo -e "  POSTGRES_PASSWORD=${POSTGRES_PASSWORD}"
echo -e "  BETTER_AUTH_SECRET=${BETTER_AUTH_SECRET}"
echo -e "  AGENT_SECRET=${AGENT_SECRET}"
echo -e "  S3_SECRET_ACCESS_KEY=${S3_SECRET}"
echo ""

# ============================================
# Step 4: OpenAI Configuration
# ============================================
echo -e "${GREEN}Step 4: OpenAI Configuration${NC}"
prompt_with_default "Enter your OpenAI API key:" "OPENAI_API_KEY" "" "no"
prompt_with_default "Enter OpenAI Base URL:" "OPENAI_BASE_URL" "$DEFAULT_OPENAI_BASE_URL" "no"
prompt_with_default "Enter OpenAI Model:" "OPENAI_MODEL" "$DEFAULT_OPENAI_MODEL" "no"

# ============================================
# Step 5: OAuth Configuration
# ============================================
echo -e "${GREEN}Step 5: OAuth Configuration (Optional)${NC}"
prompt_with_default "Enter Google OAuth Client ID:" "GOOGLE_CLIENT_ID" "" "no"

if [ -n "$GOOGLE_CLIENT_ID" ]; then
    prompt_with_default "Enter Google OAuth Client Secret:" "GOOGLE_CLIENT_SECRET" "" "no"
fi

prompt_with_default "Enter Spotify Client ID:" "SPOTIFY_CLIENT_ID" "" "no"

if [ -n "$SPOTIFY_CLIENT_ID" ]; then
    prompt_with_default "Enter Spotify Client Secret:" "SPOTIFY_CLIENT_SECRET" "" "no"
fi

# ============================================
# Step 6: Save Environment File
# ============================================
echo -e "${GREEN}Step 6: Saving Environment File${NC}"

# Ensure directory exists
mkdir -p "$AETHER_DIR"

# Build DATABASE_URL
if [ -z "$DATABASE_URL" ]; then
    DATABASE_URL="postgresql://aether:${POSTGRES_PASSWORD}@localhost:5432/aether"
fi

# Save to .env (complete config — matches docker/.env.example)
cat > "$ENV_FILE" << EOF
# =============================================================================
# REQUIRED
# =============================================================================

DOMAIN=$DOMAIN

POSTGRES_PASSWORD=$POSTGRES_PASSWORD
BETTER_AUTH_SECRET=$BETTER_AUTH_SECRET
AGENT_SECRET=$AGENT_SECRET

OPENAI_API_KEY=$OPENAI_API_KEY
OPENAI_BASE_URL=${OPENAI_BASE_URL:-$DEFAULT_OPENAI_BASE_URL}
OPENAI_MODEL=${OPENAI_MODEL:-$DEFAULT_OPENAI_MODEL}

# =============================================================================
# DEFAULTS
# =============================================================================

POSTGRES_DB=aether
POSTGRES_USER=aether
DATABASE_URL=$DATABASE_URL

BETTER_AUTH_URL=https://$DOMAIN
BETTER_AUTH_TRUSTED_ORIGINS=https://$DOMAIN

AGENT_IMAGE=suryaumapathy2812/aether-agent:latest
AGENT_NETWORK=aether_internal
AGENT_IDLE_TIMEOUT=1800
AGENT_HEALTH_TIMEOUT=30
AGENT_PORT=8000
AGENT_ASSETS_ROOT=/var/lib/aether/agents

S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=$S3_SECRET
S3_ENDPOINT=http://aether-minio:9000
S3_BUCKET=
S3_BUCKET_TEMPLATE=core-ai-media-{user}
S3_REGION=us-east-1
S3_PUBLIC_BASE_URL=
S3_FORCE_PATH_STYLE=true
S3_PUT_URL_TTL_SECONDS=300
S3_GET_URL_TTL_SECONDS=900

AGENT_UPDATE_REPO=suryaumapathy2812/aether
AGENT_UPDATE_TOKEN=
AGENT_STATE_KEY=

AETHER_AUTO_ASSIGN_FIRST_AGENT=true
AETHER_DEFAULT_AGENT_ID=

DOZZLE_PORT=8080

# =============================================================================
# OPTIONAL
# =============================================================================

VAPID_PUBLIC_KEY=
VAPID_PRIVATE_KEY=
VAPID_SUBJECT=

GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET
SPOTIFY_CLIENT_ID=$SPOTIFY_CLIENT_ID
SPOTIFY_CLIENT_SECRET=$SPOTIFY_CLIENT_SECRET

DOCKERHUB_USERNAME=$DOCKERHUB_USERNAME
EOF

echo -e "${GREEN}Environment file saved to $ENV_FILE${NC}"

# Also copy to docker directory
cp "$ENV_FILE" "$AETHER_DIR/docker/.env"
echo -e "${GREEN}Copied .env to docker directory${NC}"

# ============================================
# Step 7: Install Prerequisites
# ============================================
echo ""
echo -e "${GREEN}Step 7: Installing Prerequisites${NC}"

# Add Go to PATH (in case it was previously installed)
export PATH=$PATH:/usr/local/go/bin

# Update package list (ignore errors from third-party repos)
echo "Updating package list..."
apt update -qq 2>/dev/null || true

# Install Node.js
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
    apt install -y -qq nodejs
fi

GO_VERSION="1.26.0"

# Install Go
if ! command -v go &> /dev/null; then
    echo "Installing Go $GO_VERSION (this may take a minute)..."
    wget -q "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -O /tmp/go.tar.gz
    tar -C /usr/local -xzf /tmp/go.tar.gz
    rm /tmp/go.tar.gz
    # Add to system PATH
    echo 'export PATH=$PATH:/usr/local/go/bin' > /etc/profile.d/go.sh
    chmod +x /etc/profile.d/go.sh
    echo "Go $GO_VERSION installed!"
fi

# Add Go to PATH for this session
export PATH=$PATH:/usr/local/go/bin

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
fi

# Ensure docker is running and enabled
if command -v docker &> /dev/null; then
    systemctl enable docker 2>/dev/null || true
    systemctl start docker 2>/dev/null || true
    # Add user to docker group if not already
    if [ -n "$USER" ] && ! groups $USER | grep -q docker; then
        usermod -aG docker $USER 2>/dev/null || true
    fi
fi

# Login to Docker Hub (needed to avoid rate limits)
echo ""
echo -e "${GREEN}Docker Hub Login${NC}"
echo "Logging in to Docker Hub to avoid rate limits..."
echo "If you don't have a Docker Hub account, create one at https://hub.docker.com"
docker logout 2>/dev/null || true

# Try to login with credentials if provided, otherwise prompt
if [ -n "$DOCKERHUB_USERNAME" ] && [ -n "$DOCKERHUB_TOKEN" ]; then
    echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
elif [ -n "$DOCKERHUB_USERNAME" ]; then
    docker login -u "$DOCKERHUB_USERNAME"
fi

# Install PM2
if ! command -v pm2 &> /dev/null; then
    echo "Installing PM2..."
    npm install -g pm2
fi

# Install Docker Compose (v2 plugin or v1 standalone)
if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    # Try installing docker-compose-v2 package first (available on Ubuntu 24.04+)
    apt install -y -qq docker-compose-v2 2>/dev/null || true
    # If that didn't work, install docker-compose v1 as fallback
    if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
        curl -fsSL "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    fi
fi

# Determine docker compose command (v2 plugin vs v1)
DOCKER_COMPOSE="docker compose"
if ! command -v docker compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
fi

echo -e "${GREEN}Prerequisites installed!${NC}"

# ============================================
# Step 8: Start Docker Services
# ============================================
echo ""
echo -e "${GREEN}Step 8: Starting Docker Services${NC}"
cd "$AETHER_DIR/docker"
$DOCKER_COMPOSE -f docker-compose.services.yml up -d

# Wait for services to be healthy
echo "Waiting for Postgres..."
until docker exec aether-postgres pg_isready -U aether > /dev/null 2>&1; do
    sleep 1
done

echo -e "${GREEN}Docker services started!${NC}"

# ============================================
# Step 9: Build Dashboard
# ============================================
echo ""
echo -e "${GREEN}Step 9: Building Dashboard${NC}"
cd "$AETHER_DIR/dashboard"
npm install
npm run build

# Push Prisma schema to database (creates tables for auth, API keys, etc.)
echo "Running database migrations..."
npx prisma db push

# ============================================
# Step 10: Build Orchestrator
# ============================================
echo ""
echo -e "${GREEN}Step 10: Building Orchestrator${NC}"
cd "$AETHER_DIR/orchestrator"
go build -o aether-orchestrator ./cmd/server

# ============================================
# Step 11: Setup PM2
# ============================================
echo ""
echo -e "${GREEN}Step 11: Setting Up PM2${NC}"
cd "$AETHER_DIR/docker"

# Create log directory
mkdir -p /var/log/aether

# Pre-pull agent image so first user request is fast
AGENT_IMAGE="${AGENT_IMAGE:-suryaumapathy2812/aether-agent:latest}"
echo "Pre-pulling agent image: $AGENT_IMAGE"
docker pull "$AGENT_IMAGE"

# Load environment and start PM2
set -a
source "$ENV_FILE"
set +a

pm2 start ecosystem.config.js
pm2 save
pm2 startup

# ============================================
# Step 12: Setup Caddy
# ============================================
echo ""
echo -e "${GREEN}Step 12: Setting Up Caddy${NC}"

# Stop existing caddy if any
docker rm -f caddy 2>/dev/null || true

# Create Caddy data directory
mkdir -p /var/lib/caddy

# Run Caddy via Docker (host networking so it can reach localhost:3000/4000)
docker run -d \
    --name caddy \
    --network host \
    --restart=unless-stopped \
    -v "$AETHER_DIR/docker/Caddyfile:/etc/caddy/Caddyfile:ro" \
    -v /var/lib/caddy:/data \
    caddy:latest

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}     Setup Complete!                 ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Services:"
echo -e "  - Dashboard: http://localhost:3000"
echo -e "  - Orchestrator: http://localhost:4000"
echo -e "  - Postgres: localhost:5432"
echo -e "  - MinIO Console: http://localhost:9001"
echo -e "  - Dozzle (logs): http://localhost:8080"
echo -e "  - Caddy: https://$DOMAIN (host networking)"
echo ""
echo -e "Commands:"
echo -e "  - PM2: pm2 status, pm2 logs"
echo -e "  - Docker services: cd $AETHER_DIR/docker && $DOCKER_COMPOSE -f docker-compose.services.yml logs"
echo -e "  - Restart dashboard: pm2 restart aether-dashboard"
echo -e "  - Restart orchestrator: pm2 restart aether-orchestrator"
echo ""
echo -e "${YELLOW}IMPORTANT: Save your passwords from Step 3!${NC}"
echo ""
