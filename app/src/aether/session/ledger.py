"""
Task Ledger — P↔E communication channel.

The Task Ledger is the single communication channel between the P Worker
and E Worker, as defined in Requirements.md §2.2.1. It wraps the task
CRUD operations in SessionStore with domain-specific semantics:

  - P Worker: submit tasks (pending), read any task's status/result
  - E Worker: pick up pending tasks, set to running, set to complete/error
  - LLM: read the ledger via the check_task tool

The ledger is backed by SQLite (same DB as sessions/messages). Tasks are
never deleted — they form a complete audit trail of all work the agent
has done. Storage is negligible: thousands of tasks = kilobytes.

Status transitions are strict:
  pending → running → complete
                    → error

A failed task is retried by creating a new task, not by resetting status.

On restart, tasks that were 'running' when the agent died are re-queued
as 'pending' so the E Worker picks them up again.

Usage:
    ledger = TaskLedger(session_store)
    await ledger.resume_interrupted()  # On startup

    # P Worker submits
    task_id = await ledger.submit("sess-1", "sub_agent", {"prompt": "..."})

    # E Worker picks up
    task = await ledger.pick_next()
    await ledger.set_running(task.task_id)
    # ... do work ...
    await ledger.set_complete(task.task_id, {"result": "done"})

    # LLM queries
    tasks = await ledger.get_tasks(session_id="sess-1")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aether.session.models import Task, TaskStatus

if TYPE_CHECKING:
    from aether.session.store import SessionStore

logger = logging.getLogger(__name__)

# Priority ordering for pick_next: high before normal before low
_PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}


class TaskLedger:
    """
    P↔E communication channel backed by SQLite.

    Thin wrapper over SessionStore task operations that enforces
    domain rules: status transitions, priority ordering, restart
    recovery.
    """

    def __init__(self, session_store: "SessionStore") -> None:
        self._store = session_store

    # ─── P Worker operations ──────────────────────────────────

    async def submit(
        self,
        session_id: str,
        task_type: str,
        payload: dict[str, Any] | None = None,
        priority: str = "normal",
    ) -> str:
        """Submit a new task. Returns the task_id.

        Called by the P Worker to delegate work to the E Worker.
        The task starts in 'pending' status.
        """
        task = await self._store.create_task(
            session_id=session_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
        )
        logger.debug(
            "Task submitted: %s (type=%s, session=%s, priority=%s)",
            task.task_id,
            task_type,
            session_id,
            priority,
        )
        return task.task_id

    # ─── E Worker operations ──────────────────────────────────

    async def pick_next(
        self,
        task_type: str | None = None,
    ) -> Task | None:
        """Pick the next pending task, priority-weighted FIFO.

        Returns None if no pending tasks. Does NOT set the task to
        running — the caller must call set_running() explicitly after
        picking up the task. This allows the caller to decide whether
        to actually process it.
        """
        tasks = await self._store.get_tasks(
            status=TaskStatus.PENDING.value,
            task_type=task_type,
            limit=50,  # Fetch a batch, sort by priority
        )
        if not tasks:
            return None

        # Sort by priority (high first), then by submitted_at (oldest first)
        tasks.sort(
            key=lambda t: (
                _PRIORITY_ORDER.get(t.priority, 1),
                t.submitted_at,
            )
        )
        return tasks[0]

    async def set_running(self, task_id: str) -> None:
        """Mark a task as running. Called by E Worker after picking it up.

        Only valid transition: pending → running.
        """
        await self._store.update_task_status(task_id, TaskStatus.RUNNING.value)
        logger.debug("Task running: %s", task_id)

    async def set_complete(
        self,
        task_id: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Mark a task as complete with an optional result.

        Only valid transition: running → complete.
        """
        await self._store.update_task_status(
            task_id, TaskStatus.COMPLETE.value, result=result
        )
        logger.debug("Task complete: %s", task_id)

    async def set_error(self, task_id: str, error: str) -> None:
        """Mark a task as failed with an error message.

        Only valid transition: running → error.
        """
        await self._store.update_task_status(
            task_id, TaskStatus.ERROR.value, error=error
        )
        logger.warning("Task error: %s — %s", task_id, error)

    # ─── Query operations (P Worker + LLM) ────────────────────

    async def get_task(self, task_id: str) -> Task | None:
        """Get a single task by ID."""
        return await self._store.get_task(task_id)

    async def get_tasks(
        self,
        session_id: str | None = None,
        status: str | None = None,
        task_type: str | None = None,
        limit: int = 100,
    ) -> list[Task]:
        """Query tasks with optional filters.

        This is the method the LLM calls (via check_task tool) to
        inspect the ledger: "what tasks are running?", "is my sub-agent
        done?", "what did I do in the last hour?".
        """
        return await self._store.get_tasks(
            session_id=session_id,
            status=status,
            task_type=task_type,
            limit=limit,
        )

    # ─── Restart recovery ─────────────────────────────────────

    async def resume_interrupted(self) -> int:
        """Re-queue tasks that were running when the agent died.

        Called once on startup. Tasks with status 'running' were
        in-flight when the process crashed. They are reset to 'pending'
        so the E Worker picks them up again.

        Returns the number of tasks re-queued.
        """
        interrupted = await self._store.get_interrupted_tasks()
        count = 0
        for task in interrupted:
            await self._store.requeue_task(task.task_id)
            logger.info(
                "Re-queued interrupted task: %s (type=%s, session=%s)",
                task.task_id,
                task.type,
                task.session_id,
            )
            count += 1

        if count > 0:
            logger.info("Resumed %d interrupted tasks on startup", count)
        return count
