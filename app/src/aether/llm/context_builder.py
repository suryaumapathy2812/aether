"""
Context Builder — builds LLMRequestEnvelope with skill/plugin injection.

This is where the magic happens:
1. Build system prompt from base_style + custom_instructions
2. Inject matched skills (keyword-based)
3. Inject plugin instructions (from SKILL.md)
4. Add memory context (retrieved facts, actions, sessions)
5. Build tool schemas (built-in + plugin tools)
6. Build plugin context (OAuth tokens, configs — NOT in prompt)

The ContextBuilder ensures consistent prompt construction across all
LLM consumers (ReplyService, MemoryService, NotificationService).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aether.core.config as config_module
from aether.core.config import config
from aether.llm.contracts import LLMRequestEnvelope

if TYPE_CHECKING:
    from aether.memory.store import MemoryStore
    from aether.plugins.context import PluginContextStore
    from aether.skills.loader import SkillLoader
    from aether.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Base system prompt
SYSTEM_PROMPT_BASE = """You are Aether — a warm, thoughtful personal assistant. You speak naturally and conversationally, like a close friend who happens to be incredibly knowledgeable.

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

# Voice-specific system prompt additions — appended when mode == "voice"
VOICE_PROMPT_ADDON = """
Voice mode rules (critical):
- Start EVERY response with a very short acknowledgment (1-5 words): "Sure.", "Got it.", "Okay.", "Yeah,", "Right,", "Of course." — this gets spoken immediately while the rest generates.
- Never use markdown: no bullet points, no headers, no bold/italic, no code blocks.
- Never spell out URLs or file paths verbatim — say "the link" or "that file" instead.
- Spell out numbers naturally: "twenty three" not "23", "half past two" not "2:30".
- No lists. Convert any list to flowing prose: "You've got three options: first... second... and third..."
- Keep responses under 3 sentences unless the user explicitly asks for detail.
- No emoji. No special characters.
"""

# Style modifiers — appended to system prompt based on user preference
STYLE_MODIFIERS = {
    "default": "",
    "concise": "\nStyle: Be extremely concise. Use short sentences. Avoid filler words. Get to the point fast.",
    "detailed": "\nStyle: Be thorough and detailed. Explain your reasoning. Provide context and examples when helpful.",
    "friendly": "\nStyle: Be extra warm and casual. Use conversational language, light humor, and encouraging tone.",
    "professional": "\nStyle: Be polished and professional. Clear, structured responses. Minimal casual language.",
}


@dataclass
class SessionState:
    """Minimal session state for context building."""

    session_id: str
    user_id: str
    mode: str = "text"  # text, voice
    history: list[dict[str, Any]] = field(default_factory=list)


class ContextBuilder:
    """
    Builds LLMRequestEnvelope from various sources.

    This is where skills and plugins get injected into the LLM context.

    Injection order (matters for prompt construction):
    1. Base system prompt (from config.personality.base_style)
    2. Injected skills (matched by keyword overlap with user_message)
    3. Plugin instructions (from plugin SKILL.md files)
    4. Memory context (retrieved facts, actions, sessions)
    5. Conversation history
    6. Tool schemas (from tool_registry + plugin tools)
    7. Plugin context (OAuth tokens, configs - NOT in prompt, for tool execution)
    """

    def __init__(
        self,
        skill_loader: "SkillLoader | None" = None,
        plugin_context_store: "PluginContextStore | None" = None,
        tool_registry: "ToolRegistry | None" = None,
        memory_store: "MemoryStore | None" = None,
    ):
        self.skill_loader = skill_loader
        self.plugin_context_store = plugin_context_store
        self.tool_registry = tool_registry
        self.memory_store = memory_store

    async def build(
        self,
        user_message: str,
        session: SessionState,
        enabled_plugins: list[str] | None = None,
        pending_memory: str | None = None,
        pending_vision: dict[str, Any] | None = None,
    ) -> LLMRequestEnvelope:
        """
        Build a complete LLM request envelope.

        Args:
            user_message: The user's message
            session: Session state with history
            enabled_plugins: List of enabled plugin names
            pending_memory: Pre-retrieved memory context
            pending_vision: Vision context (image data)

        Returns:
            LLMRequestEnvelope ready for LLM Core
        """
        enabled_plugins = enabled_plugins or []

        # Auto-retrieve memory if not pre-populated
        if pending_memory is None and self.memory_store is not None:
            pending_memory = await self._retrieve_memory(user_message)

        # 1. Build system prompt
        system_prompt = self._build_system_prompt(
            user_message=user_message,
            enabled_plugins=enabled_plugins,
            mode=session.mode,
        )

        # 2. Build messages
        messages = self._build_messages(
            system_prompt=system_prompt,
            session=session,
            user_message=user_message,
            pending_memory=pending_memory,
            pending_vision=pending_vision,
        )

        # 3. Build tool schemas
        tools = self._build_tool_schemas(enabled_plugins)

        # 4. Build plugin context (for tool execution)
        plugin_context = self._build_plugin_context(enabled_plugins)

        # 5. Build policy
        policy = self._build_policy(mode=session.mode)

        return LLMRequestEnvelope(
            kind="reply_text" if session.mode == "text" else "reply_voice",
            modality=session.mode,
            user_id=session.user_id,
            session_id=session.session_id,
            messages=messages,
            tools=tools,
            plugin_context=plugin_context,
            policy=policy,
        )

    def _build_system_prompt(
        self,
        user_message: str,
        enabled_plugins: list[str],
        mode: str = "text",
    ) -> str:
        """Build the full system prompt with all injections."""
        parts = [SYSTEM_PROMPT_BASE]

        # Voice-specific rules (acknowledgment prefix, no markdown, etc.)
        if mode == "voice":
            parts.append(VOICE_PROMPT_ADDON)

        # Apply style modifier
        style = config_module.config.personality.base_style.lower()
        modifier = STYLE_MODIFIERS.get(style, "")
        if modifier:
            parts.append(modifier)

        # Custom instructions
        instructions = config_module.config.personality.custom_instructions.strip()
        if instructions:
            parts.append(
                f"\n\nUser's custom instructions (always follow these):\n{instructions}"
            )

        # Skill injection
        if self.skill_loader:
            # Always list available skills
            skills_section = self.skill_loader.get_system_prompt_section()
            if skills_section:
                parts.append("\n\n" + skills_section)

            # If a skill matches the query, inject its full instructions
            matched = self.skill_loader.match(user_message)
            if matched:
                skill = matched[0]  # Best match
                logger.info(f"Skill: {skill.name}")
                parts.append(f"\n\n--- Active Skill: {skill.name} ---\n{skill.content}")

        # Plugin instructions (from SKILL.md)
        plugin_instructions = self._get_plugin_instructions(enabled_plugins)
        if plugin_instructions:
            parts.append(f"\n\n--- Active Plugins ---{plugin_instructions}")

        return "\n".join(parts)

    def _get_plugin_instructions(self, enabled_plugins: list[str]) -> str:
        """Get plugin instructions from SKILL.md files."""
        instructions = ""
        for plugin_name in enabled_plugins:
            # Try to load plugin's SKILL.md
            skill_content = self._load_plugin_skill(plugin_name)
            if skill_content:
                instructions += f"\n\n[Plugin: {plugin_name}]\n{skill_content}"
        return instructions

    def _load_plugin_skill(self, plugin_name: str) -> str | None:
        """Load a plugin's SKILL.md content."""
        # This would normally load from the plugin directory
        # For now, return None - actual implementation would read the file
        # Path: app/plugins/{plugin_name}/SKILL.md
        try:
            from pathlib import Path

            skill_path = (
                Path(__file__).parent.parent.parent
                / "plugins"
                / plugin_name
                / "SKILL.md"
            )
            if skill_path.exists():
                return skill_path.read_text()
        except Exception as e:
            logger.debug(f"Could not load skill for {plugin_name}: {e}")
        return None

    def _build_messages(
        self,
        system_prompt: str,
        session: SessionState,
        user_message: str,
        pending_memory: str | None,
        pending_vision: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build the messages list for the LLM call."""
        messages: list[dict[str, Any]] = []

        # System prompt
        messages.append({"role": "system", "content": system_prompt})

        # Memory context
        if pending_memory:
            messages.append(
                {
                    "role": "system",
                    "content": f"Relevant context from past conversations:\n{pending_memory}\n\nUse this naturally if relevant.",
                }
            )

        # Conversation history
        max_turns = config_module.config.llm.max_history_turns
        messages.extend(session.history[-max_turns * 2 :])

        # User message (with optional vision)
        if pending_vision:
            import base64

            image_b64 = base64.b64encode(pending_vision["data"]).decode("utf-8")
            mime = pending_vision.get("mime_type", "image/jpeg")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": user_message})

        return messages

    def _build_tool_schemas(self, enabled_plugins: list[str]) -> list[dict[str, Any]]:
        """Build tool schemas from registry + plugin tools."""
        if not self.tool_registry:
            return []

        tools = []

        # Built-in tools
        for tool in self.tool_registry.list_tools():
            tools.append(tool.to_openai_schema())

        # Note: Plugin tools are already registered in the tool_registry
        # with plugin_name, so they're included above.
        # The tool_orchestrator will filter by plugin_context at execution time.

        return tools

    def _build_plugin_context(
        self, enabled_plugins: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Build plugin context for tool execution (OAuth tokens, configs)."""
        if not self.plugin_context_store:
            return {}

        context = {}
        for plugin_name in enabled_plugins:
            plugin_ctx = self.plugin_context_store.get(plugin_name)
            if plugin_ctx:
                context[plugin_name] = plugin_ctx

        return context

    def _build_policy(self, mode: str = "text") -> dict[str, Any]:
        """Build policy from config. Voice uses a faster model for lower latency."""
        cfg = config_module.config.llm
        model = cfg.voice_model if mode == "voice" else cfg.model
        return {
            "provider": cfg.provider,
            "model": model,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }

    async def _retrieve_memory(self, user_message: str) -> str | None:
        """Retrieve relevant memory context for the user message.

        Mirrors the logic from MemoryRetrieverProcessor but returns a
        formatted string instead of a Frame.
        """
        assert self.memory_store is not None
        try:
            results = await self.memory_store.search(
                user_message, limit=config.memory.search_limit
            )
            if not results:
                return None

            lines: list[str] = []
            for r in results:
                if r.get("type") == "fact":
                    lines.append(f"[Known fact] {r['fact']}")
                elif r.get("type") == "action":
                    args = r.get("arguments", "{}")
                    output_preview = r.get("output", "")[:100]
                    status = "failed" if r.get("error") else "succeeded"
                    lines.append(
                        f"[Past action] Used {r['tool_name']}({args}) — "
                        f"{status}: {output_preview}"
                    )
                elif r.get("type") == "session":
                    lines.append(f"[Previous session] {r['summary']}")
                elif r.get("type") == "conversation":
                    lines.append(
                        f"[Previous conversation] User said: {r['user_message']} — "
                        f"You replied: {r['assistant_message']}"
                    )
                else:
                    if r.get("user_message"):
                        lines.append(
                            f"[Previous conversation] User said: {r['user_message']} — "
                            f"You replied: {r['assistant_message']}"
                        )

            if lines:
                logger.info(f"Memory: {len(lines)} matches")
                return "\n".join(lines)
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}", exc_info=True)
        return None
