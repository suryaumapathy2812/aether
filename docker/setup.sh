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

# Default values (will be overridden by .env if present)
DEFAULT_DOMAIN="aether.suryaumapathy.in"
DEFAULT_OPENAI_BASE_URL="https://api.openai.com/v1"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     Aether Setup Script            ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root is not recommended.${NC}"
    echo ""
fi

# Load existing .env if present
AETHER_DIR="/opt/aether"
ENV_FILE="$AETHER_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    echo -e "${GREEN}Found existing .env file at $ENV_FILE${NC}"
    echo "Loading existing configuration..."
    set -a  # Auto-export
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
# Step 2: Security Configuration
# ============================================
echo -e "${GREEN}Step 2: Security Configuration${NC}"

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
# Step 3: OpenAI Configuration
# ============================================
echo -e "${GREEN}Step 3: OpenAI Configuration${NC}"
prompt_with_default "Enter your OpenAI API key:" "OPENAI_API_KEY" "" "no"
prompt_with_default "Enter OpenAI Base URL:" "OPENAI_BASE_URL" "$DEFAULT_OPENAI_BASE_URL" "no"

# ============================================
# Step 4: OAuth Configuration
# ============================================
echo -e "${GREEN}Step 4: OAuth Configuration (Optional)${NC}"
prompt_with_default "Enter Google OAuth Client ID:" "GOOGLE_CLIENT_ID" "" "no"

if [ -n "$GOOGLE_CLIENT_ID" ]; then
    prompt_with_default "Enter Google OAuth Client Secret:" "GOOGLE_CLIENT_SECRET" "" "no"
fi

prompt_with_default "Enter Spotify Client ID:" "SPOTIFY_CLIENT_ID" "" "no"

if [ -n "$SPOTIFY_CLIENT_ID" ]; then
    prompt_with_default "Enter Spotify Client Secret:" "SPOTIFY_CLIENT_SECRET" "" "no"
fi

# ============================================
# Step 5: Save Environment File
# ============================================
echo -e "${GREEN}Step 5: Saving Environment File${NC}"

# Ensure directory exists
mkdir -p "$AETHER_DIR"

# Build DATABASE_URL
if [ -z "$DATABASE_URL" ]; then
    DATABASE_URL="postgresql://aether:${POSTGRES_PASSWORD}@localhost:5432/aether"
fi

# Save to .env
cat > "$ENV_FILE" << EOF
# === DOMAIN ===
DOMAIN=$DOMAIN

# === DATABASE ===
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
DATABASE_URL=$DATABASE_URL

# === AUTH ===
BETTER_AUTH_SECRET=$BETTER_AUTH_SECRET

# === AGENT ===
AGENT_SECRET=$AGENT_SECRET

# === LLM ===
OPENAI_API_KEY=$OPENAI_API_KEY
OPENAI_BASE_URL=$OPENAI_BASE_URL

# === S3/MinIO ===
S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=$S3_SECRET
S3_ENDPOINT=http://localhost:9000

# === OAUTH ===
GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET
SPOTIFY_CLIENT_ID=$SPOTIFY_CLIENT_ID
SPOTIFY_CLIENT_SECRET=$SPOTIFY_CLIENT_SECRET

# === DOCKER ===
DOCKERHUB_USERNAME=$DOCKERHUB_USERNAME
EOF

echo -e "${GREEN}Environment file saved to $ENV_FILE${NC}"

# Also copy to docker directory
cp "$ENV_FILE" "$AETHER_DIR/docker/.env"
echo -e "${GREEN}Copied .env to docker directory${NC}"

# ============================================
# Step 6: Install Prerequisites
# ============================================
echo ""
echo -e "${GREEN}Step 6: Installing Prerequisites${NC}"

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

# Install Go
if ! command -v go &> /dev/null; then
    echo "Installing Go (this may take a minute)..."
    wget -q https://go.dev/dl/go1.22.linux-amd64.tar.gz -O /tmp/go.tar.gz
    tar -C /usr/local -xzf /tmp/go.tar.gz
    rm /tmp/go.tar.gz
    # Add to system PATH
    echo 'export PATH=$PATH:/usr/local/go/bin' > /etc/profile.d/go.sh
    chmod +x /etc/profile.d/go.sh
    echo "Go installed!"
fi

# Add Go to PATH for this session
export PATH=$PATH:/usr/local/go/bin

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker $USER
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

# Install Docker Compose
if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    apt install -y -qq docker-compose-v2
fi

# Determine docker compose command (v1 vs v2)
DOCKER_COMPOSE="docker compose"
if ! command -v docker compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
fi

echo -e "${GREEN}Prerequisites installed!${NC}"

# ============================================
# Step 7: Start Docker Services
# ============================================
echo ""
echo -e "${GREEN}Step 7: Starting Docker Services${NC}"
cd "$AETHER_DIR/docker"
$DOCKER_COMPOSE -f docker-compose.services.yml up -d

# Wait for services to be healthy
echo "Waiting for Postgres..."
until docker exec aether-postgres pg_isready -U aether > /dev/null 2>&1; do
    sleep 1
done

echo -e "${GREEN}Docker services started!${NC}"

# ============================================
# Step 8: Build Dashboard
# ============================================
echo ""
echo -e "${GREEN}Step 8: Building Dashboard${NC}"
cd "$AETHER_DIR/dashboard"
npm install
npm run build

# ============================================
# Step 9: Build Orchestrator
# ============================================
echo ""
echo -e "${GREEN}Step 9: Building Orchestrator${NC}"
cd "$AETHER_DIR/orchestrator"
go build -o aether-orchestrator ./cmd/server

# ============================================
# Step 10: Setup PM2
# ============================================
echo ""
echo -e "${GREEN}Step 10: Setting Up PM2${NC}"
cd "$AETHER_DIR/docker"

# Create log directory
mkdir -p /var/log/aether

# Update ecosystem.config.js with correct paths
sed -i "s|/opt/aether/dashboard|$AETHER_DIR/dashboard|g" ecosystem.config.js
sed -i "s|/opt/aether/orchestrator|$AETHER_DIR/orchestrator|g" ecosystem.config.js

# Load environment and start PM2
set -a
source "$ENV_FILE"
set +a

pm2 start ecosystem.config.js
pm2 save
pm2 startup

# ============================================
# Step 11: Setup Caddy
# ============================================
echo ""
echo -e "${GREEN}Step 11: Setting Up Caddy${NC}"

# Stop existing caddy if any
docker rm -f caddy 2>/dev/null || true

# Create Caddy data directory
mkdir -p /var/lib/caddy

# Run Caddy via Docker
docker run -d \
    --name caddy \
    -p 80:80 \
    -p 443:443 \
    --restart=unless-stopped \
    -v "$AETHER_DIR/docker/Caddyfile:/etc/caddy/Caddyfile:ro" \
    -v /var/lib/caddy:/data \
    -e CADDY_ENV=prod \
    -e DOMAIN="$DOMAIN" \
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
echo -e "  - Caddy: https://$DOMAIN"
echo ""
echo -e "Commands:"
echo -e "  - PM2: pm2 status, pm2 logs"
echo -e "  - Docker services: cd $AETHER_DIR/docker && $DOCKER_COMPOSE -f docker-compose.services.yml logs"
echo -e "  - Restart dashboard: pm2 restart aether-dashboard"
echo -e "  - Restart orchestrator: pm2 restart aether-orchestrator"
echo ""
echo -e "${YELLOW}IMPORTANT: Save your passwords from Step 2!${NC}"
echo ""
