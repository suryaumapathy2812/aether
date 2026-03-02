# Aether Dashboard - Production Dockerfile (Bun)
#
# Build:  docker build -t aether-dashboard:latest ./dashboard
# Run:    docker run -p 3000:3000 aether-dashboard:latest

# Stage 1: Install dependencies
FROM oven/bun:1 AS deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --ignore-scripts

# Stage 2: Build the Next.js application
FROM oven/bun:1 AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY prisma ./prisma
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN bun install --frozen-lockfile --ignore-scripts
RUN bunx prisma generate
RUN bun run build

# Stage 3: Production image (standalone output)
FROM oven/bun:1 AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
# Create a dedicated user for security best practices
RUN groupadd --gid 1001 nodejs && useradd --uid 1001 --gid 1001 --create-home --shell /bin/sh nextjs
RUN mkdir .next
RUN chown nextjs:bun .next
# Copy standalone output and static files from the builder stage
COPY --from=builder --chown=nextjs:bun /app/.next/standalone ./
COPY --from=builder --chown=nextjs:bun /app/public ./public
COPY --from=builder --chown=nextjs:bun /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:bun /app/src/generated/prisma ./src/generated/prisma

USER nextjs
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["bun", "server.js"]
