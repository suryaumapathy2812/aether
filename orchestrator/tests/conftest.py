"""
Shared fixtures for orchestrator tests.

Provides a mock asyncpg pool and a FastAPI TestClient with all DB
operations stubbed out — no real Postgres needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── Mock asyncpg pool ──────────────────────────────────────


class MockRecord(dict):
    """Dict subclass that supports attribute-style access like asyncpg.Record."""

    def __getitem__(self, key):
        return super().__getitem__(key)


def make_record(**kwargs) -> MockRecord:
    """Create a mock asyncpg Record."""
    return MockRecord(**kwargs)


class MockPool:
    """
    In-memory mock of asyncpg.Pool.

    Tracks all SQL calls for assertion. Returns configurable responses.
    """

    def __init__(self):
        self.execute = AsyncMock(return_value=None)
        self.fetch = AsyncMock(return_value=[])
        self.fetchrow = AsyncMock(return_value=None)
        self.fetchval = AsyncMock(return_value=1)
        self._conn = AsyncMock()
        self._conn.execute = AsyncMock()

    def acquire(self):
        """Return an async context manager that yields a mock connection."""
        return _MockAcquire(self._conn)


class _MockAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_pool():
    """A fresh MockPool for each test."""
    return MockPool()


@pytest.fixture
def patch_get_pool(mock_pool):
    """Patch get_pool() to return our mock pool."""

    async def _get_pool():
        return mock_pool

    with (
        patch("src.db.get_pool", _get_pool),
        patch("src.auth.get_pool", _get_pool),
        patch("src.main.get_pool", _get_pool),
    ):
        yield mock_pool


@pytest.fixture
def client(patch_get_pool):
    """
    FastAPI TestClient with mocked DB pool.

    Skips startup/shutdown events (no real DB or Docker).
    """
    from fastapi.testclient import TestClient
    from src.main import app

    # Disable lifespan events — they need real DB + Docker
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
