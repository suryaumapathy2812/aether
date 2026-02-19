"""Database layer — asyncpg connection pool and schema bootstrap."""

from __future__ import annotations

import os
import asyncpg

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            os.environ["DATABASE_URL"].replace(
                "postgresql+asyncpg://", "postgresql://"
            ),
            min_size=2,
            max_size=10,
        )
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def bootstrap_schema():
    """Create tables if they don't exist.

    Schema is shared with the dashboard (better-auth manages user/session/account/verification).
    The orchestrator creates the tables it owns and ensures better-auth core tables exist.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            -- ── Better Auth core tables ──
            -- These are also managed by Prisma in the dashboard,
            -- but we create them here so the orchestrator can start independently.

            CREATE TABLE IF NOT EXISTS "user" (
                id              TEXT PRIMARY KEY,
                email           TEXT UNIQUE NOT NULL,
                name            TEXT,
                email_verified  BOOLEAN DEFAULT false,
                image           TEXT,
                created_at      TIMESTAMPTZ DEFAULT now(),
                updated_at      TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS session (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL REFERENCES "user"(id),
                token           TEXT UNIQUE NOT NULL,
                expires_at      TIMESTAMPTZ NOT NULL,
                ip_address      TEXT,
                user_agent      TEXT,
                created_at      TIMESTAMPTZ DEFAULT now(),
                updated_at      TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS account (
                id                          TEXT PRIMARY KEY,
                user_id                     TEXT NOT NULL REFERENCES "user"(id),
                account_id                  TEXT NOT NULL,
                provider_id                 TEXT NOT NULL,
                access_token                TEXT,
                refresh_token               TEXT,
                access_token_expires_at     TIMESTAMPTZ,
                refresh_token_expires_at    TIMESTAMPTZ,
                scope                       TEXT,
                id_token                    TEXT,
                password                    TEXT,
                created_at                  TIMESTAMPTZ DEFAULT now(),
                updated_at                  TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS verification (
                id              TEXT PRIMARY KEY,
                identifier      TEXT NOT NULL,
                value           TEXT NOT NULL,
                expires_at      TIMESTAMPTZ NOT NULL,
                created_at      TIMESTAMPTZ DEFAULT now(),
                updated_at      TIMESTAMPTZ DEFAULT now()
            );

            -- ── Aether domain tables ──

            CREATE TABLE IF NOT EXISTS devices (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES "user"(id),
                name        TEXT DEFAULT 'Unknown Device',
                device_type TEXT DEFAULT 'ios',
                token       TEXT UNIQUE NOT NULL,
                paired_at   TIMESTAMPTZ DEFAULT now(),
                last_seen   TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS agents (
                id              TEXT PRIMARY KEY,
                user_id         TEXT UNIQUE REFERENCES "user"(id),
                container_id    TEXT,
                container_name  TEXT,
                host            TEXT NOT NULL,
                port            INTEGER NOT NULL,
                status          TEXT DEFAULT 'starting',
                keep_alive      BOOLEAN DEFAULT false,
                registered_at   TIMESTAMPTZ DEFAULT now(),
                last_health     TIMESTAMPTZ,
                created_at      TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS pair_requests (
                code        TEXT PRIMARY KEY,
                device_type TEXT DEFAULT 'ios',
                device_name TEXT DEFAULT 'Unknown Device',
                created_at  TIMESTAMPTZ DEFAULT now(),
                expires_at  TIMESTAMPTZ DEFAULT now() + interval '10 minutes',
                claimed_by  TEXT REFERENCES "user"(id)
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES "user"(id),
                provider    TEXT NOT NULL,
                key_value   TEXT NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT now(),
                UNIQUE(user_id, provider)
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id         TEXT PRIMARY KEY REFERENCES "user"(id),
                -- Voice pipeline
                stt_provider    TEXT DEFAULT 'deepgram',
                stt_model       TEXT DEFAULT 'nova-3',
                stt_language    TEXT DEFAULT 'en',
                llm_provider    TEXT DEFAULT 'openai',
                llm_model       TEXT DEFAULT 'gpt-4o',
                tts_provider    TEXT DEFAULT 'openai',
                tts_model       TEXT DEFAULT 'tts-1',
                tts_voice       TEXT DEFAULT 'nova',
                -- Personality
                base_style      TEXT DEFAULT 'default',
                custom_instructions TEXT DEFAULT '',
                -- Timestamps
                updated_at      TIMESTAMPTZ DEFAULT now()
            );

            -- ── Plugin tables ──

            CREATE TABLE IF NOT EXISTS plugins (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES "user"(id),
                name        TEXT NOT NULL,
                enabled     BOOLEAN DEFAULT false,
                installed_at TIMESTAMPTZ DEFAULT now(),
                UNIQUE(user_id, name)
            );

            CREATE TABLE IF NOT EXISTS plugin_configs (
                id          TEXT PRIMARY KEY,
                plugin_id   TEXT NOT NULL REFERENCES plugins(id) ON DELETE CASCADE,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                updated_at  TIMESTAMPTZ DEFAULT now(),
                UNIQUE(plugin_id, key)
            );

            CREATE TABLE IF NOT EXISTS plugin_events (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES "user"(id),
                plugin_name TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                source_id   TEXT,
                summary     TEXT,
                decision    TEXT DEFAULT 'pending',
                scheduled_for TIMESTAMPTZ,
                batch_notification TEXT,
                payload     JSONB DEFAULT '{}',
                created_at  TIMESTAMPTZ DEFAULT now()
            );

            -- Migration: add columns if they don't exist (for existing tables)
            ALTER TABLE plugin_events ADD COLUMN IF NOT EXISTS scheduled_for TIMESTAMPTZ;
            ALTER TABLE plugin_events ADD COLUMN IF NOT EXISTS batch_notification TEXT;
            ALTER TABLE agents ADD COLUMN IF NOT EXISTS keep_alive BOOLEAN DEFAULT false;

            CREATE INDEX IF NOT EXISTS idx_plugin_events_user
                ON plugin_events(user_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_deferred_flush
                ON plugin_events(user_id, scheduled_for)
                WHERE decision = 'deferred' AND scheduled_for IS NOT NULL;
        """)
