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

# Spinner
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while ps -p $pid > /dev/null 2>&1; do
        local temp=${spinstr#?}
        printf " [%c] " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     Aether Setup Script            ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root is not recommended.${NC}"
    echo ""
fi

# Get domain
echo -e "${GREEN}Step 1: Domain Configuration${NC}"
echo -e "Enter your domain (e.g., aether.suryaumapathy.in):"
read -r DOMAIN
if [ -z "$DOMAIN" ]; then
    echo -e "${RED}Domain is required. Exiting.${NC}"
    exit 1
fi

# Get secrets
echo ""
echo -e "${GREEN}Step 2: Security Configuration${NC}"

generate_password() {
    openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32
}

echo "Generating secure passwords..."
POSTGRES_PASSWORD=$(generate_password)
BETTER_AUTH_SECRET=$(generate_password)
AGENT_SECRET=$(generate_password)
S3_SECRET=$(generate_password)

echo -e "Passwords generated. Please save these:"
echo ""
echo -e "${YELLOW}POSTGRES_PASSWORD=${POSTGRES_PASSWORD}${NC}"
echo -e "${YELLOW}BETTER_AUTH_SECRET=${BETTER_AUTH_SECRET}${NC}"
echo -e "${YELLOW}AGENT_SECRET=${AGENT_SECRET}${NC}"
echo -e "${YELLOW}S3_SECRET_ACCESS_KEY=${S3_SECRET}${NC}"
echo ""

# Get OpenAI key
echo -e "${GREEN}Step 3: OpenAI Configuration${NC}"
echo -e "Enter your OpenAI API key (or press Enter to skip):"
read -r OPENAI_API_KEY

echo -e "Enter OpenAI Base URL (optional, press Enter for default https://api.openai.com/v1):"
read -r OPENAI_BASE_URL
if [ -z "$OPENAI_BASE_URL" ]; then
    OPENAI_BASE_URL="https://api.openai.com/v1"
fi

# Get OAuth credentials
echo ""
echo -e "${GREEN}Step 4: OAuth Configuration (Optional)${NC}"
echo -e "Enter Google OAuth Client ID (or press Enter to skip):"
read -r GOOGLE_CLIENT_ID

if [ -n "$GOOGLE_CLIENT_ID" ]; then
    echo -e "Enter Google OAuth Client Secret:"
    read -r GOOGLE_CLIENT_SECRET
fi

echo -e "Enter Spotify Client ID (or press Enter to skip):"
read -r SPOTIFY_CLIENT_ID

if [ -n "$SPOTIFY_CLIENT_ID" ]; then
    echo -e "Enter Spotify Client Secret:"
    read -r SPOTIFY_CLIENT_SECRET
fi

# Create .env file
echo ""
echo -e "${GREEN}Step 5: Creating Environment File${NC}"

cat > .env << EOF
# === DOMAIN ===
DOMAIN=$DOMAIN

# === DATABASE ===
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
DATABASE_URL=postgresql://aether:${POSTGRES_PASSWORD}@localhost:5432/aether

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
EOF

echo -e "${GREEN}Environment file created: .env${NC}"

# Install prerequisites
echo ""
echo -e "${GREEN}Step 6: Installing Prerequisites${NC}"

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
    echo "Installing Go..."
    wget -q https://go.dev/dl/go1.22.linux-amd64.tar.gz -O /tmp/go.tar.gz
    tar -C /usr/local -xzf /tmp/go.tar.gz
    rm /tmp/go.tar.gz
fi

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker $USER
fi

# Install PM2
if ! command -v pm2 &> /dev/null; then
    echo "Installing PM2..."
    npm install -g pm2
fi

# Install Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo "Installing Docker Compose..."
    apt install -y -qq docker-compose-v2
fi

echo -e "${GREEN}Prerequisites installed!${NC}"

# Clone repository (if not already in correct location)
echo ""
echo -e "${GREEN}Step 7: Setting Up Aether${NC}"

AETHER_DIR="/opt/aether"

if [ ! -d "$AETHER_DIR" ]; then
    echo "Cloning Aether repository..."
    git clone https://github.com/suryaumapathy2812/aether.git $AETHER_DIR
    cd $AETHER_DIR
else
    echo "Aether directory already exists at $AETHER_DIR"
    cd $AETHER_DIR
fi

# Create log directory
mkdir -p /var/log/aether

# Copy environment file
cp docker/.env.template docker/.env
cp docker/.env.template /opt/aether/.env

# Update .env with actual values
cat > /opt/aether/.env << EOF
DOMAIN=$DOMAIN
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
DATABASE_URL=postgresql://aether:${POSTGRES_PASSWORD}@localhost:5432/aether
BETTER_AUTH_SECRET=$BETTER_AUTH_SECRET
AGENT_SECRET=$AGENT_SECRET
OPENAI_API_KEY=$OPENAI_API_KEY
OPENAI_BASE_URL=$OPENAI_BASE_URL
S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=$S3_SECRET
S3_ENDPOINT=http://localhost:9000
GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET
SPOTIFY_CLIENT_ID=$SPOTIFY_CLIENT_ID
SPOTIFY_CLIENT_SECRET=$SPOTIFY_CLIENT_SECRET
EOF

echo -e "${GREEN}Aether setup complete!${NC}"

# Start Docker services
echo ""
echo -e "${GREEN}Step 8: Starting Docker Services${NC}"
cd $AETHER_DIR/docker
docker compose -f docker-compose.services.yml up -d

# Wait for services to be healthy
echo "Waiting for Postgres..."
until docker exec aether-postgres pg_isready -U aether > /dev/null 2>&1; do
    sleep 1
done

echo -e "${GREEN}Docker services started!${NC}"

# Build and start Dashboard
echo ""
echo -e "${GREEN}Step 9: Building Dashboard${NC}"
cd $AETHER_DIR/dashboard
npm install
npm run build

# Build and start Orchestrator
echo ""
echo -e "${GREEN}Step 10: Building Orchestrator${NC}"
cd $AETHER_DIR/orchestrator
go build -o aether-orchestrator ./cmd/server

# Setup PM2
echo ""
echo -e "${GREEN}Step 11: Setting Up PM2${NC}"
cd $AETHER_DIR/docker

# Create log directory
mkdir -p /var/log/aether

# Update ecosystem.config.js with correct paths
sed -i "s|/opt/aether/dashboard|$AETHER_DIR/dashboard|g" ecosystem.config.js
sed -i "s|/opt/aether/orchestrator|$AETHER_DIR/orchestrator|g" ecosystem.config.js

# Load environment and start PM2
export $(cat /opt/aether/.env | xargs) && pm2 start ecosystem.config.js
pm2 save
pm2 startup

# Setup Caddy
echo ""
echo -e "${GREEN}Step 12: Setting Up Caddy${NC}"

# Create Caddy data directory
mkdir -p /var/lib/caddy

# Run Caddy via Docker
docker run -d \
    --name caddy \
    -p 80:80 \
    -p 443:443 \
    --restart=unless-stopped \
    -v $AETHER_DIR/docker/Caddyfile:/etc/caddy/Caddyfile:ro \
    -v /var/lib/caddy:/data \
    -e CADDY_ENV=prod \
    -e DOMAIN=$DOMAIN \
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
echo -e "  - Docker services: cd $AETHER_DIR/docker && docker compose -f docker-compose.services.yml logs"
echo -e "  - Restart dashboard: pm2 restart aether-dashboard"
echo -e "  - Restart orchestrator: pm2 restart aether-orchestrator"
echo ""
echo -e "${YELLOW}IMPORTANT: Save your passwords from Step 4!${NC}"
echo ""
