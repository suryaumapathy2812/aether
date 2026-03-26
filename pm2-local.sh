#!/bin/bash
#
# Aether Local PM2 Manager
#
# Usage:
#   ./pm2-local.sh start    - Start all services
#   ./pm2-local.sh stop     - Stop all services
#   ./pm2-local.sh restart - Restart all services
#   ./pm2-local.sh logs    - View logs
#   ./pm2-local.sh status  - Show status
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PM2_FILE="ecosystem.config.js"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

cd "$SCRIPT_DIR"

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo -e "${RED}PM2 not found. Installing...${NC}"
    npm install -g pm2
fi

# Ensure Docker services are running
echo -e "${YELLOW}Ensuring Docker services are running...${NC}"
cd docker
docker compose -f docker-compose.services.yml up -d
cd ..

# Wait for Postgres
echo "Waiting for Postgres..."
until docker exec aether-postgres pg_isready -U aether >/dev/null 2>&1; do
    sleep 1
done
echo -e "${GREEN}Docker services ready!${NC}"

case "$1" in
    start)
        echo -e "${GREEN}Starting Aether services with PM2...${NC}"
        
        # Check for cloudflared tunnel URL
        if [ -z "$CHANNELS_WEBHOOK_URL" ]; then
            echo -e "${YELLOW}WARNING: CHANNELS_WEBHOOK_URL not set!${NC}"
            echo "For Telegram webhooks, you need to:"
            echo "  1. Start cloudflared: cloudflared tunnel --url localhost:80"
            echo "  2. Copy the URL to agent/.env as CHANNELS_WEBHOOK_URL"
            echo "  3. Restart: ./pm2-local.sh restart"
            echo ""
        fi
        
        pm2 start "$PM2_FILE"
        echo ""
        echo -e "${GREEN}All services started!${NC}"
        echo ""
        pm2 status
        ;;
        
    stop)
        echo -e "${YELLOW}Stopping Aether services...${NC}"
        pm2 stop "$PM2_FILE"
        echo -e "${GREEN}All services stopped!${NC}"
        ;;
        
    restart)
        echo -e "${YELLOW}Restarting Aether services...${NC}"
        pm2 restart "$PM2_FILE"
        echo -e "${GREEN}All services restarted!${NC}"
        ;;
        
    delete)
        echo -e "${YELLOW}Deleting all PM2 processes...${NC}"
        pm2 delete all
        echo -e "${GREEN}All processes deleted!${NC}"
        ;;
        
    logs)
        echo -e "${YELLOW}Showing logs (Ctrl+C to exit)...${NC}"
        pm2 logs
        ;;
        
    status)
        pm2 status
        ;;
        
    mon)
        echo -e "${YELLOW}Starting PM2 monit...${NC}"
        pm2 monit
        ;;
        
    build)
        echo -e "${GREEN}Building dashboard...${NC}"
        cd client/web/aether
        bash -l -c 'source ~/.vite-plus/env && vp build'
        echo -e "${GREEN}Dashboard built!${NC}"
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|delete|logs|status|mon|build}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  delete  - Delete all PM2 processes"
        echo "  logs    - View logs (live)"
        echo "  status  - Show service status"
        echo "  mon     - Show PM2 monitor"
        echo "  build   - Build dashboard for production"
        exit 1
        ;;
esac
