# Aether Dashboard (`dashboard`)

Next.js web app for users to authenticate, chat with their agent, configure plugins/services, and manage account/device settings.

## What this service owns

- Better Auth UI + `/api/auth/*` route handling
- Product pages: home, chat, agent, devices, services, plugins, memory, account
- Session token sync to orchestrator API client
- Browser-side UX for plugin settings and agent controls

## Current architecture

- App router pages: `src/app/*`
- Auth config: `src/lib/auth.ts`, `src/lib/auth-client.ts`
- Orchestrator API client: `src/lib/api.ts`
- Session bridge component: `src/components/SessionSync.tsx`
- DB/Auth schema: `prisma/schema.prisma`

## Run locally

```bash
npm install
npm run dev
```

## Build and start

```bash
npm run build
npm run start
```

## Prisma helpers

```bash
npm run db:push
npm run db:studio
```

## Build image

```bash
docker build -t aether-dashboard:local .
```

## Important env vars

- `DATABASE_URL` (Prisma + Better Auth adapter)
- `BETTER_AUTH_SECRET`, `BETTER_AUTH_URL`, `BETTER_AUTH_TRUSTED_ORIGINS`
- `NEXT_PUBLIC_ORCHESTRATOR_URL` (optional direct orchestrator base override)

## Current constraints

- Chat depends on orchestrator proxying; dashboard does not host its own chat backend route.
- API auth token is held in-memory for client calls and requires active session sync.
- No dedicated test script is defined yet in `package.json`.
