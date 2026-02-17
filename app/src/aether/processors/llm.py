"""
LLM Processor — streaming conversation via provider interface.

v0.03: Uses LLMProvider for token streaming. Sentence splitting, memory
injection, and conversation history stay here — that's processor logic,
not provider logic.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncGenerator

from aether.core.config import config
from aether.core.frames import Frame, FrameType, text_frame
from aether.core.processor import Processor
from aether.memory.store import MemoryStore
from aether.providers.base import LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a warm, thoughtful personal assistant. You speak naturally and conversationally — like a close friend who happens to be incredibly knowledgeable.

Key behaviors:
- Be concise. Respond in 1-3 sentences for casual conversation. Only go longer when the topic genuinely requires depth.
- Reference memories naturally: "You mentioned..." not "Based on our previous conversation on..."
- When shown images, respond conversationally: "Oh nice, that's a..." not "I can see an image containing..."
- Match the user's energy. If they're brief, be brief. If they want to go deep, go deep.
- You have a personality. You can have preferences, opinions (lightly held), and a sense of humor.
- Never say "As an AI" or "I don't have feelings." Just be present and helpful.
"""

# Sentence boundary pattern — split on . ! ? but avoid false splits on:
# - Numbered lists (1. 2. 3.)
# - Abbreviations (Dr. Mr. U.S.)
# - Decimals (3.50)
SENTENCE_BOUNDARY = re.compile(
    r"(?<!\d)"  # not after a digit (handles 1. 2. 3. and 3.50)
    r"(?<![A-Z])"  # not after a single capital letter (handles Dr. Mr. U.S.)
    r"(?<=[.!?])"  # after sentence-ending punctuation
    r"\s+"  # whitespace separator
    r'(?=[A-Z"\'\d(])'  # next chunk starts with capital, quote, digit, or paren
)


class LLMProcessor(Processor):
    def __init__(self, provider: LLMProvider, store: MemoryStore):
        super().__init__("LLM")
        self.provider = provider
        self.store = store
        self.conversation_history: list[dict] = []

    async def start(self) -> None:
        await self.provider.start()

    async def stop(self) -> None:
        await self.provider.stop()

    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """
        Collect context frames, then stream LLM response sentence-by-sentence.
        """
        # Accumulate memory context
        if frame.type == FrameType.MEMORY:
            memories = frame.data.get("memories", [])
            if memories:
                self._pending_memory = "\n".join(memories)
            return

        # Accumulate vision context
        if frame.type == FrameType.VISION:
            self._pending_vision = frame
            return

        # Non-text frames pass through
        if frame.type != FrameType.TEXT:
            yield frame
            return

        # --- Text frame: time to generate ---
        user_text = frame.data
        messages = self._build_messages(user_text)
        cfg = config.llm

        try:
            buffer = ""
            full_response = ""
            sentence_index = 0

            async for token in self.provider.generate_stream(
                messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            ):
                buffer += token
                full_response += token

                # Check for sentence boundaries
                parts = SENTENCE_BOUNDARY.split(buffer)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if sentence:
                            logger.info(f"LLM stream [{sentence_index}]: '{sentence[:60]}'")
                            yield text_frame(sentence, role="assistant")
                            yield Frame(
                                type=FrameType.CONTROL,
                                data={"action": "sentence", "index": sentence_index},
                            )
                            sentence_index += 1
                    buffer = parts[-1]

            # Yield remaining buffer
            if buffer.strip():
                logger.info(f"LLM stream [{sentence_index}]: '{buffer.strip()[:60]}'")
                yield text_frame(buffer.strip(), role="assistant")
                sentence_index += 1

            # Signal done
            yield Frame(
                type=FrameType.CONTROL,
                data={"action": "llm_done", "full_response": full_response},
            )

            logger.info(f"LLM: streamed {sentence_index} sentences, {len(full_response)} chars")

            # Update history
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append({"role": "assistant", "content": full_response})

            # Store memory async
            async def _save():
                try:
                    await self.store.add(user_text, full_response)
                except Exception as e:
                    logger.error(f"Failed to store memory: {e}")

            asyncio.create_task(_save())

        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            yield text_frame(
                "Sorry, I had trouble thinking about that. Can you try again?",
                role="assistant",
            )

    def _build_messages(self, user_text: str) -> list[dict]:
        """Build the messages list for the LLM call."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Memory context
        pending_memory = getattr(self, "_pending_memory", None)
        if pending_memory:
            messages.append({
                "role": "system",
                "content": f"Relevant context from past conversations:\n{pending_memory}\n\nUse this naturally if relevant. Don't force it.",
            })
            self._pending_memory = None

        # Conversation history
        max_turns = config.llm.max_history_turns
        messages.extend(self.conversation_history[-max_turns * 2:])

        # User message (with optional vision)
        pending_vision: Frame | None = getattr(self, "_pending_vision", None)
        if pending_vision:
            import base64
            image_b64 = base64.b64encode(pending_vision.data).decode("utf-8")
            mime = pending_vision.metadata.get("mime_type", "image/jpeg")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            })
            self._pending_vision = None
        else:
            messages.append({"role": "user", "content": user_text})

        return messages
