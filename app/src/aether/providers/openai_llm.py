"""
OpenAI LLM Provider — GPT-4o streaming with tool calling.

v0.05: Added generate_stream_with_tools for function calling support.
Streams tokens AND tool calls, handles the API contract.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from aether.core.config import config
from aether.providers.base import LLMProvider, LLMStreamEvent, LLMToolCall

logger = logging.getLogger(__name__)


class OpenAILLMProvider(LLMProvider):
    def __init__(self):
        self.client: AsyncOpenAI | None = None

    async def start(self) -> None:
        self.client = AsyncOpenAI()
        logger.info(f"OpenAI LLM provider ready (model: {config.llm.model})")

    async def stop(self) -> None:
        self.client = None

    async def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from OpenAI (text only, no tools)."""
        if not self.client:
            raise RuntimeError("OpenAI LLM not started")

        stream = await self.client.chat.completions.create(
            model=config.llm.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    async def generate_stream_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """
        Stream response with OpenAI function calling.

        The stream can contain text tokens AND tool calls.
        Tool calls are accumulated across chunks (OpenAI sends them incrementally),
        then yielded as a single tool_calls event when the stream finishes.
        """
        if not self.client:
            raise RuntimeError("OpenAI LLM not started")

        kwargs: dict = {
            "model": config.llm.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await self.client.chat.completions.create(**kwargs)

        # Accumulate tool calls across chunks
        # OpenAI sends tool calls incrementally: index, name, then argument chunks
        pending_tool_calls: dict[int, dict] = {}
        has_tool_calls = False

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # Text content
            if delta.content:
                yield LLMStreamEvent(type="token", content=delta.content)

            # Tool calls (accumulated across chunks)
            if delta.tool_calls:
                has_tool_calls = True
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in pending_tool_calls:
                        pending_tool_calls[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name or "" if tc.function else "",
                            "arguments": "",
                        }
                    else:
                        if tc.id:
                            pending_tool_calls[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            pending_tool_calls[idx]["name"] = tc.function.name

                    if tc.function and tc.function.arguments:
                        pending_tool_calls[idx]["arguments"] += tc.function.arguments

            # Check for finish
            if choice.finish_reason == "tool_calls" or (
                choice.finish_reason == "stop" and has_tool_calls
            ):
                # Parse accumulated tool calls
                tool_calls = []
                for idx in sorted(pending_tool_calls.keys()):
                    tc_data = pending_tool_calls[idx]
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool args: {tc_data['arguments'][:100]}")
                        args = {}

                    tool_calls.append(LLMToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args,
                    ))

                if tool_calls:
                    yield LLMStreamEvent(type="tool_calls", tool_calls=tool_calls)

        yield LLMStreamEvent(type="done")

    async def health_check(self) -> dict:
        status = "ready" if self.client else "not_started"
        return {
            "provider": "openai",
            "model": config.llm.model,
            "status": status,
        }
