// Aether PM2 Ecosystem Configuration
//
// Usage:
//   pm2 start ecosystem.config.js
//   pm2 save
//   pm2 startup
//
// Environment:
//   Load variables from .env file
//

module.exports = {
  apps: [
    // Dashboard (Next.js)
    {
      name: 'aether-dashboard',
      script: 'npm',
      args: 'start',
      cwd: '/opt/aether/dashboard',
      instances: 1,
      exec_mode: 'fork',
      env: {
        NODE_ENV: 'production',
        PORT: 3000,
        DATABASE_URL: process.env.DATABASE_URL || 'postgresql://aether:@localhost:5432/aether',
        BETTER_AUTH_SECRET: process.env.BETTER_AUTH_SECRET,
        BETTER_AUTH_URL: process.env.BETTER_AUTH_URL || 'https://aether.suryaumapathy.in',
        BETTER_AUTH_TRUSTED_ORIGINS: process.env.BETTER_AUTH_TRUSTED_ORIGINS || 'https://aether.suryaumapathy.in',
        AGENT_BASE_URL: process.env.AGENT_BASE_URL || 'http://localhost:4000',
        NEXT_PUBLIC_ORCHESTRATOR_URL: '',
        NEXT_PUBLIC_ORCHESTRATOR_WS_URL: '',
      },
      error_file: '/var/log/aether/dashboard-error.log',
      out_file: '/var/log/aether/dashboard-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      max_memory_restart: '1G',
      kill_timeout: 5000,
    },

    // Orchestrator (Go)
    {
      name: 'aether-orchestrator',
      script: './aether-orchestrator',
      cwd: '/opt/aether/orchestrator',
      instances: 1,
      exec_mode: 'fork',
      env: {
        PORT: 4000,
        DATABASE_URL: process.env.DATABASE_URL || 'postgresql://aether:@localhost:5432/aether',
        AGENT_SECRET: process.env.AGENT_SECRET,
        AGENT_IMAGE: process.env.AGENT_IMAGE || 'suryaumapathy2812/aether-agent:latest',
        AGENT_NETWORK: process.env.AGENT_NETWORK || 'aether_internal',
        AGENT_IDLE_TIMEOUT: parseInt(process.env.AGENT_IDLE_TIMEOUT) || 1800,
        AGENT_HEALTH_TIMEOUT: parseInt(process.env.AGENT_HEALTH_TIMEOUT) || 30,
        AGENT_PORT: parseInt(process.env.AGENT_PORT) || 8000,
        OPENAI_API_KEY: process.env.OPENAI_API_KEY,
        OPENAI_BASE_URL: process.env.OPENAI_BASE_URL,
        OPENAI_MODEL: process.env.OPENAI_MODEL,
        AGENT_STATE_KEY: process.env.AGENT_STATE_KEY,
        AGENT_ASSETS_ROOT: process.env.AGENT_ASSETS_ROOT || '/var/lib/aether/agents',
        VAPID_PUBLIC_KEY: process.env.VAPID_PUBLIC_KEY,
        VAPID_PRIVATE_KEY: process.env.VAPID_PRIVATE_KEY,
        VAPID_SUBJECT: process.env.VAPID_SUBJECT,
        S3_BUCKET: process.env.S3_BUCKET,
        S3_BUCKET_TEMPLATE: process.env.S3_BUCKET_TEMPLATE || 'core-ai-media-{user}',
        S3_REGION: process.env.S3_REGION || 'us-east-1',
        S3_ACCESS_KEY_ID: process.env.S3_ACCESS_KEY_ID,
        S3_SECRET_ACCESS_KEY: process.env.S3_SECRET_ACCESS_KEY,
        S3_ENDPOINT: process.env.S3_ENDPOINT || 'http://aether-minio:9000',
        S3_PUBLIC_BASE_URL: process.env.S3_PUBLIC_BASE_URL,
        S3_FORCE_PATH_STYLE: process.env.S3_FORCE_PATH_STYLE || 'true',
        S3_PUT_URL_TTL_SECONDS: parseInt(process.env.S3_PUT_URL_TTL_SECONDS) || 300,
        S3_GET_URL_TTL_SECONDS: parseInt(process.env.S3_GET_URL_TTL_SECONDS) || 900,
        AGENT_UPDATE_REPO: process.env.AGENT_UPDATE_REPO,
        AGENT_UPDATE_TOKEN: process.env.AGENT_UPDATE_TOKEN,
        AETHER_AUTO_ASSIGN_FIRST_AGENT: process.env.AETHER_AUTO_ASSIGN_FIRST_AGENT || 'true',
        AETHER_DEFAULT_AGENT_ID: process.env.AETHER_DEFAULT_AGENT_ID,
        GOOGLE_CLIENT_ID: process.env.GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET: process.env.GOOGLE_CLIENT_SECRET,
        SPOTIFY_CLIENT_ID: process.env.SPOTIFY_CLIENT_ID,
        SPOTIFY_CLIENT_SECRET: process.env.SPOTIFY_CLIENT_SECRET,
      },
      error_file: '/var/log/aether/orchestrator-error.log',
      out_file: '/var/log/aether/orchestrator-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      kill_timeout: 10000,
    },
  ],
};
