# Dashboard/Agent Contract

This document locks the API contract orchestrator must satisfy for current
dashboard + agent features.

## Auth Model

- All dashboard requests include bearer session token.
- Orchestrator resolves identity from:
  1. `Authorization: Bearer <token>`
  2. Better Auth cookies
  3. `?token=` query
- User routing always uses token-derived `user_id`.

## Required HTTP groups

- `/v1/*` (chat, media upload, agent tasks/jobs)
- `/api/memory/*`
- `/api/plugins*`
- `/api/push/vapid-key`, `/api/push/subscribe`
- `/health`, `/api/health`
- `/api/agent/ready`
- `/api/webrtc/offer`, `/api/webrtc/ice`
- `/api/metrics/latency`

## Required WebSocket group

- `/api/ws/notifications` (or `/api/ws`) authenticated endpoint that proxies
  user-scoped notification stream to the assigned agent.
