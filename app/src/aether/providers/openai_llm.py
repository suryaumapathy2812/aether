"""
OpenAI LLM Provider — GPT-4o streaming.

Streams tokens, handles the API contract. Sentence splitting stays in the
processor layer — the provider just yields raw tokens.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from aether.core.config import config
from aether.providers.base import LLMProvider

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
        """Stream response tokens from OpenAI."""
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

    async def health_check(self) -> dict:
        status = "ready" if self.client else "not_started"
        return {
            "provider": "openai",
            "model": config.llm.model,
            "status": status,
        }
