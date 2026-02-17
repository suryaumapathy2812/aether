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
    """Create tables if they don't exist."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                email       TEXT UNIQUE NOT NULL,
                name        TEXT,
                password    TEXT NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS devices (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES users(id),
                name        TEXT DEFAULT 'Unknown Device',
                device_type TEXT DEFAULT 'ios',
                token       TEXT UNIQUE NOT NULL,
                paired_at   TIMESTAMPTZ DEFAULT now(),
                last_seen   TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS agents (
                id              TEXT PRIMARY KEY,
                user_id         TEXT UNIQUE REFERENCES users(id),
                container_id    TEXT,
                host            TEXT NOT NULL,
                port            INTEGER NOT NULL,
                status          TEXT DEFAULT 'running',
                registered_at   TIMESTAMPTZ DEFAULT now(),
                last_health     TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS pair_requests (
                code        TEXT PRIMARY KEY,
                device_type TEXT DEFAULT 'ios',
                device_name TEXT DEFAULT 'Unknown Device',
                created_at  TIMESTAMPTZ DEFAULT now(),
                expires_at  TIMESTAMPTZ DEFAULT now() + interval '10 minutes',
                claimed_by  TEXT REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES users(id),
                provider    TEXT NOT NULL,
                key_value   TEXT NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT now(),
                UNIQUE(user_id, provider)
            );
        """)
