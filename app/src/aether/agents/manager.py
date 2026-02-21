"""
Sub-Agent Manager — spawn, monitor, and retrieve results from sub-agents.

Sub-agents are independent SessionLoop instances running as asyncio.Tasks.
Each gets its own session (child of the parent), its own message history,
and its own tool access (with recursive spawning disabled).

Usage:
    manager = SubAgentManager(session_store, llm_core, context_builder, ...)
    child_id = await manager.spawn("Analyze the codebase", parent_session_id="sess-1")
    # ... later ...
    status = await manager.get_status(child_id)
    result = await manager.get_result(child_id)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from aether.agents.agent_types import get_agent_type
from aether.session.models import SessionStatus

if TYPE_CHECKING:
    from aether.kernel.event_bus import EventBus
    from aether.llm.context_builder import ContextBuilder
    from aether.llm.core import LLMCore
    from aether.session.store import SessionStore

logger = logging.getLogger(__name__)

# Default limits for sub-agents
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_MAX_DURATION = 300  # 5 minutes


class SubAgentManager:
    """
    Manages sub-agent lifecycle: spawn, monitor, cancel, get results.

    Sub-agents run as background asyncio.Tasks with their own SessionLoop.
    The parent agent continues immediately after spawning — no blocking.
    """

    def __init__(
        self,
        session_store: "SessionStore",
        llm_core: "LLMCore",
        context_builder: "ContextBuilder",
        event_bus: "EventBus | None" = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_duration: float = DEFAULT_MAX_DURATION,
    ) -> None:
        self._session_store = session_store
        self._llm_core = llm_core
        self._context_builder = context_builder
        self._event_bus = event_bus
        self._max_iterations = max_iterations
        self._max_duration = max_duration

        # Active sub-agent tasks: child_session_id → asyncio.Task
        self._tasks: dict[str, asyncio.Task] = {}
        # Abort events: child_session_id → asyncio.Event
        self._aborts: dict[str, asyncio.Event] = {}

    async def spawn(
        self,
        prompt: str,
        parent_session_id: str,
        agent_type: str = "general",
        enabled_plugins: list[str] | None = None,
    ) -> str:
        """
        Spawn a sub-agent. Returns the child session_id immediately.

        The sub-agent runs in the background as an asyncio.Task.
        The parent can check status and retrieve results later.
        """
        from aether.session.loop import SessionLoop

        child_id = f"sub-{uuid.uuid4().hex[:8]}"

        # Create child session linked to parent
        await self._session_store.create_session(
            session_id=child_id,
            agent_type=agent_type,
            parent_session_id=parent_session_id,
        )

        # Add the user prompt as the first message
        await self._session_store.append_user_message(child_id, prompt)

        # Resolve agent-type limits (agent type may override defaults)
        agent_def = get_agent_type(agent_type)
        effective_max_iter = min(self._max_iterations, agent_def.max_iterations)
        effective_max_dur = min(self._max_duration, agent_def.max_duration)

        # Create the session loop for this sub-agent
        loop = SessionLoop(
            session_store=self._session_store,
            llm_core=self._llm_core,
            context_builder=self._context_builder,
            event_bus=self._event_bus,
            max_iterations=effective_max_iter,
            max_duration=effective_max_dur,
            agent_type_override=agent_type,
        )

        abort = asyncio.Event()
        self._aborts[child_id] = abort

        task = asyncio.create_task(
            loop.run(child_id, abort=abort, enabled_plugins=enabled_plugins or [])
        )
        self._tasks[child_id] = task

        # Clean up when done and publish completion event
        task.add_done_callback(lambda t: self._on_complete(child_id))

        logger.info(
            "Spawned sub-agent %s (parent=%s, type=%s)",
            child_id,
            parent_session_id,
            agent_type,
        )
        return child_id

    async def get_status(self, session_id: str) -> dict[str, Any]:
        """Get the status of a sub-agent."""
        session = await self._session_store.get_session(session_id)
        if session is None:
            return {"status": "not_found", "session_id": session_id}

        task = self._tasks.get(session_id)
        running = task is not None and not task.done() if task else False

        return {
            "session_id": session_id,
            "status": session.status,
            "agent_type": session.agent_type,
            "parent_session_id": session.parent_session_id,
            "running": running,
        }

    async def get_result(self, session_id: str) -> str | None:
        """
        Get the final result text of a completed sub-agent.

        Returns None if the sub-agent is still running or not found.
        """
        task = self._tasks.get(session_id)
        if task and not task.done():
            return None  # Still running

        return await self._session_store.get_last_assistant_text(session_id)

    async def cancel(self, session_id: str) -> bool:
        """Cancel a running sub-agent. Returns True if canceled."""
        abort = self._aborts.get(session_id)
        if abort:
            abort.set()

        task = self._tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            logger.info("Sub-agent %s canceled", session_id)
            return True
        return False

    async def list_children(self, parent_session_id: str) -> list[dict[str, Any]]:
        """List all sub-agents for a parent session."""
        children = await self._session_store.get_child_sessions(parent_session_id)
        results = []
        for child in children:
            task = self._tasks.get(child.session_id)
            running = task is not None and not task.done() if task else False
            results.append(
                {
                    "session_id": child.session_id,
                    "status": child.status,
                    "agent_type": child.agent_type,
                    "running": running,
                }
            )
        return results

    def active_count(self) -> int:
        """Number of currently running sub-agents."""
        return sum(1 for t in self._tasks.values() if not t.done())

    async def cancel_all(self) -> int:
        """Cancel all running sub-agents. Returns count canceled."""
        canceled = 0
        for session_id in list(self._tasks.keys()):
            if await self.cancel(session_id):
                canceled += 1
        return canceled

    def _on_complete(self, session_id: str) -> None:
        """Callback when a sub-agent task finishes."""
        task = self._tasks.get(session_id)
        if task and task.done():
            try:
                exc = task.exception()
                if exc:
                    logger.error("Sub-agent %s failed: %s", session_id, exc)
            except asyncio.CancelledError:
                logger.info("Sub-agent %s was canceled", session_id)

        # Publish completion event
        if self._event_bus:
            asyncio.get_event_loop().call_soon(
                asyncio.ensure_future,
                self._event_bus.publish(
                    "task.completed",
                    {"session_id": session_id},
                ),
            )

        # Clean up abort event
        self._aborts.pop(session_id, None)

        logger.info("Sub-agent %s completed", session_id)
