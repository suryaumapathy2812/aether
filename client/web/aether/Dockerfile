# Aether Dashboard - Production Dockerfile (Node + pnpm + TanStack Start)
#
# Build:  docker build -t aether-dashboard:latest .
# Run:    docker run -p 3000:3000 aether-dashboard:latest

# Stage 1: Install dependencies
FROM node:22-slim AS deps
RUN corepack enable && corepack prepare pnpm@10.33.0 --activate
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile --ignore-scripts

# Stage 2: Build
FROM node:22-slim AS builder
RUN corepack enable && corepack prepare pnpm@10.33.0 --activate
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NODE_ENV=production
RUN DATABASE_URL="postgresql://placeholder" pnpm exec prisma generate
RUN pnpm run build

# Stage 3: Production image
FROM node:22-slim AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.output .output
COPY --from=builder /app/public ./public

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", ".output/server/index.mjs"]
