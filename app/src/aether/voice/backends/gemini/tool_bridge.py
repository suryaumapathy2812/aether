"""Tool bridge between Gemini function calls and Aether ToolRegistry."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aether.core.metrics import metrics

if TYPE_CHECKING:
    from aether.session.ledger import TaskLedger
    from aether.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


ExtraToolHandler = Callable[[dict[str, Any]], Awaitable[str | dict[str, Any]]]

_TOOL_TIMEOUTS_SECONDS: dict[str, float] = {
    "memory": 5.0,
    "file": 5.0,
    "search": 15.0,
    "web_fetch": 15.0,
    "external_api": 30.0,
    "code": 60.0,
    "browser": 90.0,
    "default": 15.0,
}


class ToolBridge:
    """Exposes tools to Gemini and executes them with TaskLedger audit logs."""

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        task_ledger: "TaskLedger",
        session_id: str = "voice",
    ) -> None:
        self._tool_registry = tool_registry
        self._task_ledger = task_ledger
        self._session_id = session_id
        self._extra_declarations: list[dict[str, Any]] = []
        self._extra_handlers: dict[str, ExtraToolHandler] = {}

    def register_extra(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: ExtraToolHandler,
    ) -> None:
        self._extra_declarations = [
            declaration
            for declaration in self._extra_declarations
            if declaration.get("name") != name
        ]
        self._extra_declarations.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
        self._extra_handlers[name] = handler

    def get_declarations(self) -> list[dict[str, Any]]:
        declarations: list[dict[str, Any]] = []

        for tool in self._tool_registry.list_tools():
            schema = tool.to_openai_schema()
            function_schema = schema.get("function", {})
            name = function_schema.get("name")
            if not name:
                continue
            declarations.append(
                {
                    "name": name,
                    "description": function_schema.get("description", ""),
                    "parameters": function_schema.get("parameters", {"type": "object"}),
                }
            )

        declarations.extend(self._extra_declarations)
        return declarations

    async def execute(
        self,
        name: str,
        arguments: str | dict[str, Any] | None,
        *,
        session_id: str | None = None,
    ) -> str:
        effective_session_id = session_id or self._session_id
        parsed_args = _parse_arguments(arguments)
        start_time = time.monotonic()
        task_id: str | None = None

        try:
            task_id = await self._task_ledger.submit(
                session_id=effective_session_id,
                task_type="tool_call",
                payload={
                    "tool_name": name,
                    "arguments": parsed_args,
                    "source": "gemini_p_worker",
                },
                priority="normal",
            )
            await self._task_ledger.set_running(task_id)
            timeout_seconds = _resolve_timeout_seconds(name)

            if name in self._extra_handlers:
                result = await asyncio.wait_for(
                    self._extra_handlers[name](parsed_args),
                    timeout=timeout_seconds,
                )
                response = _normalize_result(result)
            else:
                tool_result = await asyncio.wait_for(
                    self._tool_registry.dispatch(name, parsed_args),
                    timeout=timeout_seconds,
                )
                response = {
                    "ok": not tool_result.error,
                    "output": tool_result.output,
                    "metadata": tool_result.metadata,
                }

            duration_ms = int((time.monotonic() - start_time) * 1000)
            metrics.observe(
                "voice.tool_bridge.duration_ms",
                duration_ms,
                labels={"tool": name},
            )
            if response.get("ok", True):
                metrics.inc("voice.tool_bridge.ok", labels={"tool": name})
                await self._task_ledger.set_complete(
                    task_id,
                    {
                        "result": json.dumps(response)[:4000],
                        "duration_ms": duration_ms,
                    },
                )
            else:
                metrics.inc("voice.tool_bridge.error", labels={"tool": name})
                await self._task_ledger.set_error(
                    task_id,
                    str(
                        response.get("error") or response.get("output") or "tool failed"
                    ),
                )

            return json.dumps(response)
        except asyncio.TimeoutError:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            timeout_seconds = _resolve_timeout_seconds(name)
            metrics.observe(
                "voice.tool_bridge.duration_ms",
                duration_ms,
                labels={"tool": name},
            )
            metrics.inc("voice.tool_bridge.timeout", labels={"tool": name})
            if task_id is not None:
                try:
                    await self._task_ledger.set_error(
                        task_id,
                        f"Tool timed out after {timeout_seconds:.1f}s",
                    )
                except Exception:
                    logger.debug("Failed to mark tool timeout", exc_info=True)
            return json.dumps(
                {
                    "ok": False,
                    "error": f"Tool timed out after {timeout_seconds:.1f}s",
                    "timeout_seconds": timeout_seconds,
                }
            )
        except Exception as exc:
            logger.exception("Tool execution failed: %s", name)
            metrics.inc("voice.tool_bridge.exception", labels={"tool": name})
            if task_id is not None:
                try:
                    await self._task_ledger.set_error(task_id, str(exc))
                except Exception:
                    logger.debug("Failed to mark tool task as error", exc_info=True)
            return json.dumps({"ok": False, "error": str(exc)})


def _parse_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_result(result: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, dict):
        if "ok" not in result:
            return {"ok": True, **result}
        return result

    if not result:
        return {"ok": True, "output": ""}

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return {"ok": True, "output": result}

    if isinstance(parsed, dict):
        if "ok" in parsed:
            return parsed
        return {"ok": True, **parsed}

    return {"ok": True, "output": parsed}


def _resolve_timeout_seconds(tool_name: str) -> float:
    normalized = tool_name.strip().lower()
    if any(k in normalized for k in ("memory", "fact", "decision")):
        return _TOOL_TIMEOUTS_SECONDS["memory"]
    if any(k in normalized for k in ("file", "read_file", "write_file")):
        return _TOOL_TIMEOUTS_SECONDS["file"]
    if "search" in normalized:
        return _TOOL_TIMEOUTS_SECONDS["search"]
    if any(k in normalized for k in ("fetch", "crawl", "scrape")):
        return _TOOL_TIMEOUTS_SECONDS["web_fetch"]
    if any(k in normalized for k in ("run_command", "python", "shell", "code")):
        return _TOOL_TIMEOUTS_SECONDS["code"]
    if any(k in normalized for k in ("browser", "playwright", "navigate")):
        return _TOOL_TIMEOUTS_SECONDS["browser"]
    if any(
        k in normalized
        for k in (
            "gmail",
            "calendar",
            "drive",
            "spotify",
            "weather",
            "wolfram",
            "wikipedia",
        )
    ):
        return _TOOL_TIMEOUTS_SECONDS["external_api"]
    return _TOOL_TIMEOUTS_SECONDS["default"]
