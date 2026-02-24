"""Tests for Session API endpoints."""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aether.http.sessions import create_session_router
from aether.kernel.event_bus import EventBus
from aether.llm.contracts import LLMEventEnvelope, LLMRequestEnvelope
from aether.session.ledger import TaskLedger
from aether.session.store import SessionStore


# ─── Fixtures ─────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        s = SessionStore(db_path=db_path)
        await s.start()
        yield s
        await s.stop()


@pytest.fixture
def event_bus():
    return EventBus()


def _make_text_events(text: str) -> list[LLMEventEnvelope]:
    return [
        LLMEventEnvelope.text_chunk("req", "job", text, sequence=0),
        LLMEventEnvelope.stream_end("req", "job", sequence=1),
    ]


def _make_mock_agent(store):
    """Create a mock AgentCore with the methods the router needs."""
    agent = MagicMock()
    agent.get_active_session_loops = MagicMock(return_value=[])
    agent.run_session = AsyncMock(return_value="test-session")
    agent.cancel_session_loop = AsyncMock(return_value=False)
    return agent


@pytest.fixture
def client(store, event_bus):
    """Create a test client with the session router."""
    app = FastAPI()
    agent = _make_mock_agent(store)
    ledger = TaskLedger(store)

    router = create_session_router(
        agent=agent,
        session_store=store,
        event_bus=event_bus,
        task_ledger=ledger,
    )
    app.include_router(router)

    # We need to run startup to initialize the store
    # but the store is already started via the fixture
    return TestClient(app), agent, store


# ─── Session CRUD Tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_create_session(client):
    test_client, agent, store = client

    response = test_client.post(
        "/v1/sessions",
        json={"session_id": "my-session", "agent_type": "general"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["session_id"] == "my-session"
    assert data["agent_type"] == "general"


@pytest.mark.asyncio
async def test_create_session_auto_id(client):
    test_client, agent, store = client

    response = test_client.post("/v1/sessions", json={})
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0


@pytest.mark.asyncio
async def test_list_sessions(client):
    test_client, agent, store = client

    # Create some sessions
    await store.create_session("sess-1")
    await store.create_session("sess-2")

    response = test_client.get("/v1/sessions")
    assert response.status_code == 200
    data = response.json()
    assert len(data["sessions"]) >= 2


@pytest.mark.asyncio
async def test_get_session_status(client):
    test_client, agent, store = client

    await store.create_session("sess-1")

    response = test_client.get("/v1/sessions/sess-1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sess-1"
    assert data["status"] == "idle"
    assert data["running"] is False


@pytest.mark.asyncio
async def test_get_session_status_not_found(client):
    test_client, agent, store = client

    response = test_client.get("/v1/sessions/nonexistent/status")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_session_messages(client):
    test_client, agent, store = client

    await store.create_session("sess-1")
    await store.append_user_message("sess-1", "Hello")
    await store.append_assistant_message("sess-1", "Hi there!")

    response = test_client.get("/v1/sessions/sess-1/messages")
    assert response.status_code == 200
    data = response.json()
    assert data["message_count"] == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_get_session_messages_not_found(client):
    test_client, agent, store = client

    response = test_client.get("/v1/sessions/nonexistent/messages")
    assert response.status_code == 404


# ─── Agent Loop Control Tests ────────────────────────────────


@pytest.mark.asyncio
async def test_start_session_prompt(client):
    test_client, agent, store = client

    response = test_client.post(
        "/v1/sessions/sess-1/prompt",
        json={"message": "Do something"},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert "events_url" in data

    agent.run_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_session_prompt_missing_message(client):
    test_client, agent, store = client

    response = test_client.post(
        "/v1/sessions/sess-1/prompt",
        json={},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_cancel_session(client):
    test_client, agent, store = client

    agent.cancel_session_loop = AsyncMock(return_value=True)

    response = test_client.post("/v1/sessions/sess-1/cancel")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "canceled"


@pytest.mark.asyncio
async def test_cancel_session_not_running(client):
    test_client, agent, store = client

    response = test_client.post("/v1/sessions/nonexistent/cancel")
    assert response.status_code == 404


# ─── Tasks Endpoint Tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_list_tasks_empty(client):
    """Empty Task Ledger returns empty list."""
    test_client, agent, store = client

    response = test_client.get("/v1/tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["tasks"] == []
    assert data["count"] == 0


@pytest.mark.asyncio
async def test_list_tasks_with_ledger_entries(store, event_bus):
    """Tasks submitted to the Task Ledger appear in GET /v1/tasks."""
    app = FastAPI()
    agent = _make_mock_agent(store)
    ledger = TaskLedger(store)

    router = create_session_router(
        agent=agent,
        session_store=store,
        event_bus=event_bus,
        task_ledger=ledger,
    )
    app.include_router(router)

    # Create the session first (FK constraint on tasks.session_id)
    await store.create_session("sess-1")

    # Submit a task to the ledger
    task_id = await ledger.submit(
        session_id="sess-1",
        task_type="sub_agent",
        payload={"instruction": "Do something"},
        priority="normal",
    )

    test_client = TestClient(app)
    response = test_client.get("/v1/tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    assert any(t["task_id"] == task_id for t in data["tasks"])
