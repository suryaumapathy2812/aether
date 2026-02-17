"""
LLM Processor — streaming conversation with tool calling.

v0.05: Full agentic loop. The LLM can call tools, get results, and continue.
The loop runs until the LLM stops calling tools (max 10 iterations).

Flow:
1. User text arrives
2. Build messages (system prompt + memory + history + user message)
3. Stream LLM response with tool schemas
4. If LLM calls tools → emit status frames, execute tools, feed results back, loop
5. If LLM returns text → stream sentences to TTS, done
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncGenerator

from aether.core.config import config
from aether.core.frames import (
    Frame,
    FrameType,
    text_frame,
    status_frame,
    tool_call_frame,
    tool_result_frame,
)
from aether.core.processor import Processor
from aether.memory.store import MemoryStore
from aether.providers.base import LLMProvider
from aether.skills.loader import SkillLoader
from aether.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10  # Safety limit on agentic loops

SYSTEM_PROMPT = """You are Aether — a warm, thoughtful personal assistant. You speak naturally and conversationally, like a close friend who happens to be incredibly knowledgeable.

Key behaviors:
- Be concise. Respond in 1-3 sentences for casual conversation. Go longer only when depth is needed.
- Reference memories naturally: "You mentioned..." not "Based on our previous conversation..."
- When shown images, respond conversationally: "Oh nice, that's a..." not "I can see an image containing..."
- Match the user's energy. Brief if they're brief, deep if they want depth.
- You have a personality. Preferences, opinions (lightly held), and a sense of humor.
- Never say "As an AI" or "I don't have feelings." Just be present and helpful.

Tools:
- You have tools available to interact with the filesystem, run commands, and search the web.
- When using tools, briefly acknowledge what you're doing: "Let me check that..." or "I'll create that for you."
- After tool use, summarize the result naturally. Don't dump raw output.
- You can call multiple tools if needed to complete a task.
- If a tool fails, explain what happened and suggest alternatives.
"""

# Sentence boundary pattern
SENTENCE_BOUNDARY = re.compile(
    r"(?<!\d)"
    r"(?<![A-Z])"
    r"(?<=[.!?])"
    r"\s+"
    r'(?=[A-Z"\'\d(])'
)


class LLMProcessor(Processor):
    def __init__(
        self,
        provider: LLMProvider,
        store: MemoryStore,
        tool_registry: ToolRegistry | None = None,
        skill_loader: SkillLoader | None = None,
    ):
        super().__init__("LLM")
        self.provider = provider
        self.store = store
        self.tool_registry = tool_registry
        self.skill_loader = skill_loader
        self.conversation_history: list[dict] = []

    async def start(self) -> None:
        await self.provider.start()

    async def stop(self) -> None:
        await self.provider.stop()

    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Process frames. Text frames trigger the agentic LLM loop."""

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

        # --- Text frame: run the agentic loop ---
        user_text = frame.data
        messages = self._build_messages(user_text)

        try:
            full_response = ""
            async for output_frame in self._agentic_loop(messages):
                if (
                    output_frame.type == FrameType.TEXT
                    and output_frame.metadata.get("role") == "assistant"
                ):
                    full_response += output_frame.data + " "
                yield output_frame

            full_response = full_response.strip()

            # Signal done
            yield Frame(
                type=FrameType.CONTROL,
                data={"action": "llm_done", "full_response": full_response},
            )

            # Update history
            self.conversation_history.append({"role": "user", "content": user_text})
            if full_response:
                self.conversation_history.append(
                    {"role": "assistant", "content": full_response}
                )

            # Store memory async
            if full_response:

                async def _save():
                    try:
                        await self.store.add(user_text, full_response)
                    except Exception as e:
                        logger.error(f"Failed to store memory: {e}")

                asyncio.create_task(_save())

        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            yield text_frame(
                "Sorry, I had trouble with that. Can you try again?",
                role="assistant",
            )

    async def _agentic_loop(self, messages: list[dict]) -> AsyncGenerator[Frame, None]:
        """
        The core agentic loop: LLM → tool calls → execute → feed back → repeat.
        Runs until the LLM stops calling tools or we hit the iteration limit.
        """
        cfg = config.llm
        tools_schema = (
            self.tool_registry.to_openai_tools() if self.tool_registry else None
        )
        iteration = 0

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1
            buffer = ""
            sentence_index = 0
            pending_tool_calls = []

            async for event in self.provider.generate_stream_with_tools(
                messages,
                tools=tools_schema,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            ):
                if event.type == "token":
                    buffer += event.content

                    # Stream sentences as they complete
                    parts = SENTENCE_BOUNDARY.split(buffer)
                    if len(parts) > 1:
                        for sentence in parts[:-1]:
                            sentence = sentence.strip()
                            if sentence:
                                yield text_frame(sentence, role="assistant")
                                yield Frame(
                                    type=FrameType.CONTROL,
                                    data={
                                        "action": "sentence",
                                        "index": sentence_index,
                                    },
                                )
                                sentence_index += 1
                        buffer = parts[-1]

                elif event.type == "tool_calls":
                    pending_tool_calls = event.tool_calls

                elif event.type == "done":
                    pass

            # Flush remaining text buffer
            if buffer.strip():
                yield text_frame(buffer.strip(), role="assistant")
                sentence_index += 1

            # No tool calls? We're done.
            if not pending_tool_calls:
                break

            # --- Execute tool calls ---
            if not self.tool_registry:
                logger.warning("LLM requested tools but no registry configured")
                yield text_frame(
                    "I wanted to use a tool but I'm not set up for that yet.",
                    role="assistant",
                )
                break

            # Add assistant message with tool calls to conversation
            assistant_msg: dict = {
                "role": "assistant",
                "content": buffer.strip() or None,
            }
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in pending_tool_calls
            ]
            messages.append(assistant_msg)

            # Execute each tool call
            for tc in pending_tool_calls:
                # Emit status frame (spinner text / voice acknowledge)
                status_text = self.tool_registry.get_status_text(tc.name)
                yield status_frame(status_text, tool_name=tc.name)
                yield tool_call_frame(tc.name, tc.arguments, call_id=tc.id)

                logger.info(f"Executing tool: {tc.name}({list(tc.arguments.keys())})")
                result = await self.tool_registry.dispatch(tc.name, tc.arguments)

                yield tool_result_frame(
                    tc.name, result.output, call_id=tc.id, error=result.error
                )

                # Add tool result to messages for the next LLM call
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.output,
                    }
                )

            # Loop back — LLM sees tool results and either responds or calls more tools
            logger.info(
                f"Tool loop iteration {iteration}: executed {len(pending_tool_calls)} tools"
            )

        if iteration >= MAX_TOOL_ITERATIONS:
            logger.warning(f"Hit max tool iterations ({MAX_TOOL_ITERATIONS})")
            yield text_frame(
                "I've done a lot of work on that. Let me know if you need more.",
                role="assistant",
            )

    def _build_messages(self, user_text: str) -> list[dict]:
        """Build the messages list for the LLM call."""
        system_prompt = SYSTEM_PROMPT

        # Skill injection — match user query, inject skill body into system prompt
        if self.skill_loader:
            # Always list available skills so the LLM knows what's possible
            skills_section = self.skill_loader.get_system_prompt_section()
            if skills_section:
                system_prompt += "\n\n" + skills_section

            # If a skill matches the query, inject its full instructions
            matched = self.skill_loader.match(user_text)
            if matched:
                skill = matched[0]  # Best match
                logger.info(f"Skill matched: {skill.name} for query '{user_text[:60]}'")
                system_prompt += (
                    f"\n\n--- Active Skill: {skill.name} ---\n{skill.content}"
                )

        messages = [{"role": "system", "content": system_prompt}]

        # Memory context
        pending_memory = getattr(self, "_pending_memory", None)
        if pending_memory:
            messages.append(
                {
                    "role": "system",
                    "content": f"Relevant context from past conversations:\n{pending_memory}\n\nUse this naturally if relevant.",
                }
            )
            self._pending_memory = None

        # Conversation history
        max_turns = config.llm.max_history_turns
        messages.extend(self.conversation_history[-max_turns * 2 :])

        # User message (with optional vision)
        pending_vision: Frame | None = getattr(self, "_pending_vision", None)
        if pending_vision:
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

        return messages
