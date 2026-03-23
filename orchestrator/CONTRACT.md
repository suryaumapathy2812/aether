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

- `/agent/v1/*` (chat, media upload, memory, plugins, push, channels, jobs)
- `/go/v1/agent/ready`
- `/go/v1/agent/subdomain`
- `/go/v1/metrics/latency`
- `/go/v1/devices*`
- `/go/v1/pair/*`
- `/health`, `/api/health`, `/go/v1/health`

## Required WebSocket group

- `/agent/v1/ws/notifications` authenticated endpoint that proxies
  user-scoped notification stream to the assigned agent.
- `/agent/v1/ws/conversation` authenticated endpoint that proxies user-scoped
  conversation runtime traffic to the assigned agent.
