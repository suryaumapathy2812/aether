"""
Session API — async task endpoints + SSE event streaming.

Provides REST endpoints for managing sessions and observing
agent work in real-time via Server-Sent Events.

Endpoints:
    POST /v1/sessions                  → Create a new session
    POST /v1/sessions/{id}/prompt      → Start agent loop (202 Accepted)
    GET  /v1/sessions/{id}/events      → SSE stream of session events
    GET  /v1/sessions/{id}/status      → Current session status
    GET  /v1/sessions/{id}/messages    → All messages in session
    POST /v1/sessions/{id}/cancel      → Cancel running session
    GET  /v1/sessions                  → List sessions
    GET  /v1/tasks                     → List sub-agent tasks
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from aether.agent import AgentCore
    from aether.agents.manager import SubAgentManager
    from aether.kernel.event_bus import EventBus
    from aether.session.store import SessionStore

logger = logging.getLogger(__name__)


def create_session_router(
    agent: "AgentCore",
    session_store: "SessionStore",
    event_bus: "EventBus",
    sub_agent_manager: "SubAgentManager | None" = None,
) -> APIRouter:
    """Create the session management router."""

    router = APIRouter(prefix="/v1", tags=["sessions"])

    # ─── Session CRUD ─────────────────────────────────────────

    @router.post("/sessions")
    async def create_session(request: Request) -> JSONResponse:
        """Create a new session."""
        body = await request.json()
        session_id = body.get("session_id", str(uuid.uuid4()))
        agent_type = body.get("agent_type", "default")

        session = await session_store.create_session(
            session_id=session_id,
            agent_type=agent_type,
        )

        return JSONResponse(
            {
                "session_id": session.session_id,
                "status": session.status,
                "agent_type": session.agent_type,
                "created_at": session.created_at,
            },
            status_code=201,
        )

    @router.get("/sessions")
    async def list_sessions(
        status: str | None = None,
        limit: int = 50,
    ) -> JSONResponse:
        """List sessions, optionally filtered by status."""
        sessions = await session_store.list_sessions(status=status, limit=limit)
        return JSONResponse(
            {
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "status": s.status,
                        "agent_type": s.agent_type,
                        "parent_session_id": s.parent_session_id,
                        "created_at": s.created_at,
                        "updated_at": s.updated_at,
                    }
                    for s in sessions
                ]
            }
        )

    @router.get("/sessions/{session_id}/status")
    async def get_session_status(session_id: str) -> JSONResponse:
        """Get current session status."""
        session = await session_store.get_session(session_id)
        if session is None:
            return JSONResponse(
                {"error": f"Session {session_id} not found"}, status_code=404
            )

        active_loops = agent.get_active_session_loops()
        return JSONResponse(
            {
                "session_id": session.session_id,
                "status": session.status,
                "agent_type": session.agent_type,
                "running": session_id in active_loops,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            }
        )

    @router.get("/sessions/{session_id}/messages")
    async def get_session_messages(
        session_id: str,
        limit: int | None = None,
    ) -> JSONResponse:
        """Get all messages in a session."""
        session = await session_store.get_session(session_id)
        if session is None:
            return JSONResponse(
                {"error": f"Session {session_id} not found"}, status_code=404
            )

        messages = await session_store.get_messages_as_openai(session_id, limit=limit)
        return JSONResponse(
            {
                "session_id": session_id,
                "message_count": len(messages),
                "messages": messages,
            }
        )

    # ─── Agent Loop Control ───────────────────────────────────

    @router.post("/sessions/{session_id}/prompt")
    async def start_session_prompt(session_id: str, request: Request) -> JSONResponse:
        """
        Start the agent loop for a session. Returns 202 Accepted immediately.

        The agent runs in the background. Monitor via SSE at
        /v1/sessions/{id}/events or poll /v1/sessions/{id}/status.
        """
        body = await request.json()
        message = body.get("message", "")
        if not message:
            return JSONResponse({"error": "Missing 'message' field"}, status_code=400)

        enabled_plugins = body.get("enabled_plugins")

        try:
            await agent.run_session(
                session_id=session_id,
                user_message=message,
                enabled_plugins=enabled_plugins,
                background=True,
            )
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        return JSONResponse(
            {
                "session_id": session_id,
                "status": "accepted",
                "events_url": f"/v1/sessions/{session_id}/events",
            },
            status_code=202,
        )

    @router.post("/sessions/{session_id}/cancel")
    async def cancel_session(session_id: str) -> JSONResponse:
        """Cancel a running session loop."""
        canceled = await agent.cancel_session_loop(session_id)
        if canceled:
            return JSONResponse({"session_id": session_id, "status": "canceled"})
        return JSONResponse(
            {"session_id": session_id, "status": "not_running"},
            status_code=404,
        )

    # ─── SSE Event Stream ─────────────────────────────────────

    @router.get("/sessions/{session_id}/events")
    async def session_events(session_id: str) -> StreamingResponse:
        """
        SSE stream of all events for a session.

        Streams text chunks, tool calls, tool results, status updates,
        and completion events in real-time.
        """
        return StreamingResponse(
            _sse_generator(session_id, event_bus),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ─── Sub-Agent Tasks ──────────────────────────────────────

    @router.get("/tasks")
    async def list_tasks(
        parent_session_id: str | None = None,
    ) -> JSONResponse:
        """List sub-agent tasks."""
        if sub_agent_manager is None:
            return JSONResponse({"tasks": []})

        if parent_session_id:
            tasks = await sub_agent_manager.list_children(parent_session_id)
        else:
            # List all sessions that are sub-agents (have parent)
            all_sessions = await session_store.list_sessions(limit=100)
            tasks = [
                {
                    "session_id": s.session_id,
                    "status": s.status,
                    "agent_type": s.agent_type,
                    "parent_session_id": s.parent_session_id,
                }
                for s in all_sessions
                if s.parent_session_id is not None
            ]

        return JSONResponse({"tasks": tasks})

    @router.get("/tasks/{task_id}/status")
    async def get_task_status(task_id: str) -> JSONResponse:
        """Get sub-agent task status."""
        if sub_agent_manager is None:
            return JSONResponse(
                {"error": "Sub-agent manager not available"}, status_code=404
            )

        status = await sub_agent_manager.get_status(task_id)
        if status["status"] == "not_found":
            return JSONResponse({"error": f"Task {task_id} not found"}, status_code=404)

        # Include result if done
        result = None
        if status["status"] in ("done", "error"):
            result = await sub_agent_manager.get_result(task_id)

        return JSONResponse({**status, "result": result})

    return router


# ─── SSE Helpers ──────────────────────────────────────────────


async def _sse_generator(
    session_id: str, event_bus: "EventBus"
) -> AsyncGenerator[str, None]:
    """Generate SSE events from the EventBus for a session."""
    event_topic = f"session.{session_id}.event"
    status_topic = f"session.{session_id}.status"

    event_queue = event_bus.subscribe(event_topic)
    status_queue = event_bus.subscribe(status_topic)

    try:
        # Merge both queues into a single SSE stream
        # We use asyncio.wait to listen on both
        import asyncio

        event_done = False
        status_done = False

        while not (event_done and status_done):
            tasks = []
            if not event_done:
                tasks.append(asyncio.create_task(event_queue.get(), name="event"))
            if not status_done:
                tasks.append(asyncio.create_task(status_queue.get(), name="status"))

            if not tasks:
                break

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()

            for task in done:
                result = task.result()

                # Check for end sentinel
                if result is None:
                    if task.get_name() == "event":
                        event_done = True
                    else:
                        status_done = True
                    continue

                # Format as SSE
                event_name = task.get_name()
                yield f"event: {event_name}\ndata: {json.dumps(result)}\n\n"

        # Send done event
        yield f"event: done\ndata: {json.dumps({'session_id': session_id})}\n\n"

    finally:
        event_bus.unsubscribe(event_topic, event_queue)
        event_bus.unsubscribe(status_topic, status_queue)
