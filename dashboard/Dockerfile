# Aether Dashboard - Production Dockerfile
#
# Build:  docker build -t aether-dashboard:latest ./dashboard
# Run:    docker run -p 3000:3000 aether-dashboard:latest

# ── Base ──
FROM oven/bun:1-alpine AS base
WORKDIR /app

# ── Dependencies ──
FROM base AS deps
COPY package.json bun.lock ./
# Skip postinstall (prisma generate) during deps install - will run after schema is copied
RUN bun install --frozen-lockfile --ignore-scripts

# ── Builder ──
FROM base AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY prisma ./prisma
COPY . .

RUN bunx prisma generate
RUN bun run build

# ── Production ──
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy standalone build
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# Copy generated Prisma client (Prisma 7 custom output)
COPY --from=builder /app/src/generated/prisma ./src/generated/prisma

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
