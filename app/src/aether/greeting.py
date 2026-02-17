"""
Aether Session Greeting — the first thing you hear when you "pick up the phone".

Loads facts and recent conversations from memory, then generates a short,
personalized greeting via the LLM. First-time users get a warm intro.
Returning users get a "welcome back" that references something Aether knows.
"""

from __future__ import annotations

import logging

from aether.memory.store import MemoryStore
from aether.providers.base import LLMProvider

logger = logging.getLogger(__name__)

GREETING_PROMPT_FIRST_TIME = """You are Aether, a voice assistant. First time meeting this user. Say hi in 3-5 words MAX. Examples: "Hey, nice to meet you!", "Hi there!", "Hello, welcome!"

Greeting:"""

GREETING_PROMPT_RETURNING = """You are Aether, a voice assistant. Greet this returning user in 3-8 words MAX. Use their name if you know it. If you know what they did last time, you can briefly reference it. Keep it snappy.

Examples: "Hey Surya!", "Welcome back!", "Hi again — ready to continue?"

Known facts:
{facts}

{sessions_context}
Greeting:"""


async def generate_greeting(
    memory: MemoryStore,
    llm_provider: LLMProvider,
) -> str:
    """Generate a personalized session greeting."""
    try:
        facts = await memory.get_facts()

        if not facts:
            # First time user
            prompt = GREETING_PROMPT_FIRST_TIME
        else:
            # Returning user — include session summaries for continuity
            facts_text = "\n".join(f"- {f}" for f in facts[:5])

            sessions_context = ""
            try:
                sessions = await memory.get_session_summaries(limit=2)
                if sessions:
                    session_lines = [f"- {s['summary']}" for s in sessions]
                    sessions_context = "Recent sessions:\n" + "\n".join(session_lines)
            except Exception:
                pass

            prompt = GREETING_PROMPT_RETURNING.format(
                facts=facts_text, sessions_context=sessions_context
            )

        # Generate greeting — collect all tokens
        greeting = ""
        async for token in llm_provider.generate_stream(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15,
            temperature=0.8,
        ):
            greeting += token

        greeting = greeting.strip().strip('"')
        logger.info(f"Session greeting: '{greeting[:80]}'")
        return greeting

    except Exception as e:
        logger.error(f"Greeting generation failed: {e}")
        return "Hey there, good to see you."
