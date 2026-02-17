"""
LLM Processor — OpenAI GPT-4o for conversation and vision.

Takes text + memory + optional vision frames, yields text frame with response.
Manages conversation history within a session and stores to memory after response.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from aether.core.frames import Frame, FrameType, text_frame
from aether.core.processor import Processor
from aether.memory.store import MemoryStore

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

# Max conversation turns to keep in context (avoid token overflow)
MAX_HISTORY_TURNS = 20


class LLMProcessor(Processor):
    def __init__(self, store: MemoryStore):
        super().__init__("LLM")
        self.client = AsyncOpenAI()
        self.store = store
        self.conversation_history: list[dict] = []

    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """
        Collect text, memory, and vision frames, then generate response.

        The pipeline sends frames in order: memory_frame, vision_frame, text_frame.
        We accumulate context until we get a user text frame, then generate.
        """
        # Accumulate memory context
        if frame.type == FrameType.MEMORY:
            memories = frame.data.get("memories", [])
            if memories:
                memory_text = "\n".join(memories)
                # Inject as a system-level context note
                self._pending_memory = memory_text
            return  # Don't yield — wait for the text frame

        # Accumulate vision context
        if frame.type == FrameType.VISION:
            self._pending_vision = frame
            return  # Don't yield — wait for the text frame

        # Non-text frames pass through
        if frame.type != FrameType.TEXT:
            yield frame
            return

        # --- We have a text frame. Time to generate. ---
        user_text = frame.data

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject memory context if available
        pending_memory = getattr(self, "_pending_memory", None)
        if pending_memory:
            messages.append(
                {
                    "role": "system",
                    "content": f"Relevant context from past conversations:\n{pending_memory}\n\nUse this naturally in your response if relevant. Don't force it.",
                }
            )
            self._pending_memory = None

        # Add conversation history
        messages.extend(self.conversation_history[-MAX_HISTORY_TURNS * 2 :])

        # Build user message (text + optional vision)
        pending_vision: Frame | None = getattr(self, "_pending_vision", None)
        if pending_vision:
            # Multimodal message with image
            import base64

            image_b64 = base64.b64encode(pending_vision.data).decode("utf-8")
            mime = pending_vision.metadata.get("mime_type", "image/jpeg")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        },
                    ],
                }
            )
            self._pending_vision = None
        else:
            messages.append({"role": "user", "content": user_text})

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )

            assistant_text = response.choices[0].message.content or ""
            logger.info(f"LLM: '{assistant_text[:80]}'")

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_text}
            )

            # Store in memory (fire and forget — don't block the response)
            import asyncio

            async def _save_memory():
                try:
                    await self.store.add(user_text, assistant_text)
                except Exception as e:
                    logger.error(f"Failed to store memory: {e}")

            asyncio.create_task(_save_memory())

            yield text_frame(assistant_text, role="assistant")

        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            yield text_frame(
                "Sorry, I had trouble thinking about that. Can you try again?",
                role="assistant",
            )
