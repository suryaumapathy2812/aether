#!/usr/bin/env python3
"""
Aether TUI — Terminal User Interface client.

Full Rich-based TUI with persistent layout, styled panels,
live-updating tool spinners, and async prompt_toolkit input.

Usage: uv run aether_tui.py [--host localhost] [--port 8000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

try:
    import websockets
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.markup import escape
    from rich.columns import Columns
    from rich.rule import Rule
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.history import InMemoryHistory
except ImportError:
    print("Missing dependencies. Run from client/tui/ with:")
    print("  uv sync && uv run aether_tui.py")
    sys.exit(1)


# ── Braille spinner ──────────────────────────────────────────────────────────
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class ToolCall:
    """Tracks a single tool call lifecycle."""

    def __init__(self, call_id: str, name: str):
        self.call_id = call_id
        self.name = name
        self.started = time.monotonic()
        self.done = False
        self.error = False
        self.output = ""

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


class AetherTUI:
    """Rich-based terminal UI for Aether."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.console = Console()
        self.ws: Any = None

        # Conversation state
        self.messages: list[dict[str, str]] = []
        self.current_response = ""
        self.status_text = ""
        self.is_waiting = False

        # Tool tracking
        self.tool_calls: dict[str, ToolCall] = {}  # call_id -> ToolCall
        self.tool_order: list[str] = []  # ordered call_ids

        # Spinner
        self._spinner_idx = 0
        self._tick_task: asyncio.Task | None = None

        # Response gate
        self._response_done = asyncio.Event()

        # Live display
        self._live: Live | None = None

    # ── Connection ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        uri = f"ws://{self.host}:{self.port}/ws"
        try:
            self.ws = await websockets.connect(uri)
        except Exception as e:
            self.console.print(f"[bold red]Connection failed:[/bold red] {e}")
            sys.exit(1)

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render(self) -> Group:
        """Build the live display for the current turn only.

        Past messages are already printed permanently by _print_last_exchange(),
        so we only render the in-progress tool calls, streaming response, and status.
        """
        parts: list[Any] = []

        # Active tool calls
        if self.tool_order:
            parts.append(Text(""))
            for cid in self.tool_order:
                tc = self.tool_calls.get(cid)
                if not tc:
                    continue
                parts.append(self._render_tool(tc))

        # Streaming response (not yet committed to messages)
        if self.current_response and self.is_waiting:
            parts.append(Text(""))
            parts.append(Text(self.current_response, style="cyan"))

        # Status line
        if self.status_text and self.is_waiting:
            spin = SPINNER[self._spinner_idx % len(SPINNER)]
            parts.append(Text(""))
            parts.append(Text(f"  {spin} {self.status_text}", style="dim magenta"))

        return Group(*parts) if parts else Group(Text(""))

    def _render_tool(self, tc: ToolCall) -> Panel:
        """Render a single tool call as a bordered panel."""
        elapsed = f"{tc.elapsed:.1f}s"

        if tc.done and not tc.error:
            icon = "[green]✓[/green]"
            border = "green"
        elif tc.done and tc.error:
            icon = "[red]✗[/red]"
            border = "red"
        else:
            spin = SPINNER[self._spinner_idx % len(SPINNER)]
            icon = f"[cyan]{spin}[/cyan]"
            border = "cyan"

        # Title
        title = f"{icon} {tc.name} [dim]({elapsed})[/dim]"

        # Body: result preview
        body = ""
        if tc.done and tc.output:
            preview = tc.output[:200].replace("\n", " ↵ ")
            if len(tc.output) > 200:
                preview += " …"
            body = preview

        return Panel(
            Text.from_markup(f"[dim]{escape(body)}[/dim]") if body else Text(""),
            title=title,
            title_align="left",
            border_style=border,
            padding=(0, 1),
            expand=True,
        )

    def _refresh(self) -> None:
        """Trigger a Live display refresh."""
        if self._live:
            self._live.update(self._render())

    # ── Spinner tick ──────────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        """Advance spinner and refresh display while waiting."""
        try:
            while True:
                await asyncio.sleep(0.08)
                self._spinner_idx = (self._spinner_idx + 1) % len(SPINNER)
                if self.is_waiting:
                    self._refresh()
        except asyncio.CancelledError:
            pass

    # ── WebSocket message handler ─────────────────────────────────────────

    async def _receive_loop(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                self._handle(msg)
        except websockets.ConnectionClosed:
            self.console.print("\n[bold red]Connection lost.[/bold red]")
        except asyncio.CancelledError:
            pass

    def _handle(self, msg: dict) -> None:
        msg_type = msg.get("type")

        if msg_type == "text_chunk":
            text = msg["data"]
            sep = " " if self.current_response else ""
            self.current_response += sep + text
            self._refresh()

        elif msg_type == "status":
            self.status_text = msg["data"]
            self._refresh()

        elif msg_type == "tool_call":
            data = json.loads(msg["data"])
            cid = data.get("call_id", f"tc-{time.monotonic()}")
            tc = ToolCall(call_id=cid, name=data["name"])
            self.tool_calls[cid] = tc
            self.tool_order.append(cid)
            self._refresh()

        elif msg_type == "tool_result":
            data = json.loads(msg["data"])
            # Find matching tool call
            cid = data.get("call_id", "")
            tc = self.tool_calls.get(cid)
            if not tc:
                # Fallback: match by name
                for c in reversed(self.tool_order):
                    t = self.tool_calls[c]
                    if t.name == data["name"] and not t.done:
                        tc = t
                        break
            if tc:
                tc.done = True
                tc.error = data.get("error", False)
                tc.output = data.get("output", "")
            self._refresh()

        elif msg_type == "stream_end":
            # Commit response to messages
            if self.current_response:
                self.messages.append(
                    {"role": "assistant", "content": self.current_response}
                )
            self.current_response = ""
            self.status_text = ""
            self.is_waiting = False
            self._refresh()
            self._response_done.set()

        elif msg_type == "transcript":
            if not msg.get("interim"):
                # Voice transcript — show as user message
                self.messages.append({"role": "user", "content": msg["data"]})
                self._refresh()

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.connect()

        # Header
        self.console.print()
        self.console.print(
            Panel(
                "[bold cyan]Aether[/bold cyan] v0.06 — Terminal Client\n"
                "Type a message and press Enter. [bold]/quit[/bold] to exit.",
                border_style="dim",
                padding=(0, 1),
            )
        )
        self.console.print()

        # Start background tasks
        recv_task = asyncio.create_task(self._receive_loop())
        self._tick_task = asyncio.create_task(self._tick_loop())

        # Prompt session with history
        session: PromptSession = PromptSession(
            history=InMemoryHistory(),
        )

        try:
            while True:
                # Render current state above the prompt
                self._live = Live(
                    self._render(),
                    console=self.console,
                    refresh_per_second=12,
                    transient=True,
                )

                # Get user input
                try:
                    with patch_stdout():
                        user_input = await session.prompt_async(
                            [("class:prompt", "you → ")],
                            style=_prompt_style(),
                        )
                except (EOFError, KeyboardInterrupt):
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input in ("/quit", "/exit", "/q"):
                    break

                # Record and send
                self.messages.append({"role": "user", "content": user_input})
                self.current_response = ""
                self.tool_calls.clear()
                self.tool_order.clear()
                self.status_text = ""
                self.is_waiting = True
                self._response_done.clear()

                await self.ws.send(json.dumps({"type": "text", "data": user_input}))

                # Show live display while waiting for response
                with self._live:
                    await self._response_done.wait()

                self._live = None

                # Print final state (persisted after Live closes)
                self._print_last_exchange()

        except Exception as e:
            self.console.print(f"\n[bold red]Error:[/bold red] {e}")
        finally:
            recv_task.cancel()
            if self._tick_task:
                self._tick_task.cancel()
            if self.ws:
                await self.ws.close()
            self.console.print("\n[dim]Goodbye.[/dim]")

    def _print_last_exchange(self) -> None:
        """Print the completed exchange permanently after Live closes."""
        # Tool calls
        for cid in self.tool_order:
            tc = self.tool_calls.get(cid)
            if not tc:
                continue
            self.console.print(self._render_tool(tc))

        # Assistant response
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.console.print()
            self.console.print(Text(self.messages[-1]["content"], style="cyan"))
            self.console.print()

        # Clear tool state for next turn
        self.tool_calls.clear()
        self.tool_order.clear()


def _prompt_style():
    """prompt_toolkit style for the input prompt."""
    from prompt_toolkit.styles import Style

    return Style.from_dict(
        {
            "prompt": "#888888",
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Aether TUI Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    tui = AetherTUI(host=args.host, port=args.port)
    try:
        asyncio.run(tui.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
