from __future__ import annotations

import asyncio
import json

import pytest

from aether.voice.backends.gemini.tool_bridge import (
    ToolBridge,
    _resolve_timeout_seconds,
)


class _FakeToolRegistry:
    def list_tools(self) -> list[object]:
        return []

    async def dispatch(self, name: str, args: dict[str, object]) -> object:
        raise RuntimeError(f"unexpected dispatch: {name} {args}")


class _FakeTaskLedger:
    def __init__(self) -> None:
        self.errors: list[str] = []

    async def submit(self, **_: object) -> str:
        return "task-1"

    async def set_running(self, task_id: str) -> None:
        assert task_id == "task-1"

    async def set_complete(self, task_id: str, result: dict[str, object]) -> None:
        raise RuntimeError(f"unexpected completion: {task_id} {result}")

    async def set_error(self, task_id: str, error: str) -> None:
        assert task_id == "task-1"
        self.errors.append(error)


def test_resolve_timeout_by_tool_name_category() -> None:
    assert _resolve_timeout_seconds("memory_lookup") == 0.1
    assert _resolve_timeout_seconds("read_file") == 0.2
    assert _resolve_timeout_seconds("semantic_search") == 3.0
    assert _resolve_timeout_seconds("web_fetch_url") == 5.0
    assert _resolve_timeout_seconds("python_runner") == 30.0
    assert _resolve_timeout_seconds("browser_navigate") == 60.0
    assert _resolve_timeout_seconds("gmail_send") == 10.0
    assert _resolve_timeout_seconds("totally_custom") == 10.0


@pytest.mark.asyncio
async def test_execute_returns_timeout_error_for_slow_memory_tool() -> None:
    ledger = _FakeTaskLedger()
    bridge = ToolBridge(_FakeToolRegistry(), ledger, session_id="voice-1")

    async def _slow_handler(_: dict[str, object]) -> str:
        await asyncio.sleep(0.2)
        return json.dumps({"ok": True})

    bridge.register_extra(
        name="memory_lookup",
        description="slow memory tool",
        parameters={"type": "object"},
        handler=_slow_handler,
    )

    raw = await bridge.execute("memory_lookup", {"query": "x"})
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["timeout_seconds"] == 0.1
    assert "timed out" in payload["error"]
    assert ledger.errors
