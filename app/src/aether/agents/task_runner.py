"""
Task Runner — blocking sub-agent delegation via SubAgentManager.

Wraps SubAgentManager to provide a blocking interface: spawns a sub-agent,
waits for it to complete, and returns the result. This preserves the
original run_task tool behavior (blocks until done, max 60s timeout)
while using the new SessionLoop-based sub-agent infrastructure.

The old TaskRunner ran its own LLM loop with direct provider calls.
This version delegates to SubAgentManager, which uses SessionLoop
(persistent state, event bus, agent types, tool filtering — all wired).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aether.agents.manager import SubAgentManager

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60  # seconds
POLL_INTERVAL = 0.1  # seconds between status checks


class TaskRunner:
    """
    Runs sub-agent tasks with blocking semantics.

    Usage:
        runner = TaskRunner(sub_agent_manager)
        result = await runner.run("Create a project structure with 5 files")
    """

    def __init__(
        self,
        sub_agent_manager: "SubAgentManager",
        parent_session_id: str = "",
    ):
        self._manager = sub_agent_manager
        self._parent_session_id = parent_session_id

    async def run(
        self,
        prompt: str,
        timeout: float = DEFAULT_TIMEOUT,
        agent_type: str = "general",
    ) -> str:
        """
        Run a sub-agent task. Blocks until complete. Returns the result text.

        Spawns a sub-agent via SubAgentManager, then polls for completion
        up to the timeout. If the sub-agent doesn't finish in time, it's
        canceled and a timeout message is returned.
        """
        start = time.time()
        logger.info("TaskRunner: starting task: %s", prompt[:80])

        try:
            child_id = await self._manager.spawn(
                prompt=prompt,
                parent_session_id=self._parent_session_id,
                agent_type=agent_type,
            )

            # Poll for completion
            while True:
                elapsed = time.time() - start
                if elapsed > timeout:
                    await self._manager.cancel(child_id)
                    logger.warning(
                        "TaskRunner: task %s timed out after %.1fs", child_id, elapsed
                    )
                    return (
                        f"Task timed out after {timeout}s. "
                        "Partial work may have been completed."
                    )

                status = await self._manager.get_status(child_id)
                session_status = status.get("status", "")

                if session_status == "done":
                    result = await self._manager.get_result(child_id)
                    duration = time.time() - start
                    logger.info(
                        "TaskRunner: task %s completed in %.1fs", child_id, duration
                    )
                    return result or "(no output)"

                if session_status == "error":
                    result = await self._manager.get_result(child_id)
                    logger.error("TaskRunner: task %s failed", child_id)
                    return f"Task failed: {result or '(unknown error)'}"

                if session_status == "canceled":
                    return "Task was canceled."

                if not status.get("running", False) and session_status not in (
                    "busy",
                    "idle",
                ):
                    # Task finished with unexpected status
                    result = await self._manager.get_result(child_id)
                    return result or f"Task ended with status: {session_status}"

                await asyncio.sleep(POLL_INTERVAL)

        except asyncio.CancelledError:
            logger.info("TaskRunner: task canceled externally")
            return "Task was canceled."

        except Exception as e:
            logger.error("TaskRunner: task failed: %s", e, exc_info=True)
            return f"Task failed: {e}"
