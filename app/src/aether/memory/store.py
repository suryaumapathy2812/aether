"""
Aether Memory Store — SQLite + OpenAI embeddings.

v0.03: Two-tier memory (conversations + facts)
v0.07: Four-tier memory:
1. conversations — raw exchanges (user said X, assistant said Y)
2. facts — extracted knowledge ("user's name is Surya", "user prefers Python")
3. actions — tool calls and results ("created hello-world/main.py at 3pm Tuesday")
4. sessions — session summaries for cross-session continuity
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import aiosqlite
import numpy as np
from openai import AsyncOpenAI

import aether.core.config as config_module

logger = logging.getLogger(__name__)

FACT_EXTRACTION_PROMPT = """You are Aether's long-term memory extractor.

Goal: store only facts that improve future assistance for a "Jarvis/second-brain" assistant.

Extract ONLY durable, user-specific, decision-relevant facts from this turn:
- Identity and profile: name, role, location, timezone, recurring schedule
- Durable preferences: communication style, coding/workflow/tool preferences
- Ongoing projects, goals, commitments, deadlines
- Stable constraints: budget, device/platform limits, security/privacy boundaries
- Important relationships and recurring contacts (only when clearly stated)

Do NOT extract:
- Small talk, greetings, jokes, filler
- Temporary mood unless it implies a stable preference
- Assistant claims or advice as facts
- One-off details with no future value
- Duplicates or near-duplicates of existing memory wording

Write strict concise fact strings:
- One fact per string
- Third-person style, starting with "User ..." or "User's ..."
- Canonical and specific (avoid vague language)
- Keep each fact short (about 6-18 words)

Return ONLY a JSON array of strings.
If no high-value durable facts are present, return [] exactly.

Conversation:
User: {user_message}
Assistant: {assistant_message}

Facts (JSON array):"""


class MemoryStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or Path(config_module.config.memory.db_path)
        self.openai: AsyncOpenAI | None = None
        self._openai_base_url: str | None = None
        self._openai_api_key: str | None = None
        self._db: aiosqlite.Connection | None = None

    async def start(self) -> None:
        """Initialize the database and create tables."""
        self._db = await aiosqlite.connect(str(self.db_path))

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL,
                embedding TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)

        # Facts table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL UNIQUE,
                fact_key TEXT,
                embedding TEXT NOT NULL,
                source_conversation_id INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # v0.07: Actions table — tool calls and results
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                arguments TEXT NOT NULL,
                output TEXT NOT NULL,
                error BOOLEAN DEFAULT FALSE,
                embedding TEXT NOT NULL,
                session_id TEXT,
                timestamp REAL NOT NULL
            )
        """)

        # v0.07: Sessions table — cross-session continuity
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                embedding TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL NOT NULL,
                turns INTEGER NOT NULL,
                tools_used TEXT
            )
        """)

        await self._db.commit()

        await self._migrate_fact_keys()

        # Compact old actions on startup
        await self._compact_old_actions()

        logger.info(f"Memory store initialized at {self.db_path}")

    async def stop(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def add(self, user_message: str, assistant_message: str) -> None:
        """Store a conversation turn and extract facts."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        # Store the raw conversation
        combined = f"User: {user_message}\nAssistant: {assistant_message}"
        embedding = await self._embed(combined)

        cursor = await self._db.execute(
            "INSERT INTO conversations (user_message, assistant_message, embedding, timestamp) VALUES (?, ?, ?, ?)",
            (user_message, assistant_message, json.dumps(embedding), time.time()),
        )
        conv_id = int(cursor.lastrowid or 0)
        await self._db.commit()
        logger.debug(f"Stored conversation: {user_message[:50]}...")

        # Extract and store facts (fire and forget — don't block the pipeline)
        try:
            await self._extract_facts(user_message, assistant_message, conv_id)
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")

    async def _extract_facts(
        self, user_message: str, assistant_message: str, conv_id: int
    ) -> None:
        """Use LLM to extract facts from a conversation turn."""
        prompt = FACT_EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_message=assistant_message,
        )

        try:
            response = await self._get_openai_client().chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap for extraction
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )

            content_raw = response.choices[0].message.content or ""
            content = content_raw.strip()

            # Parse JSON array
            if content.startswith("["):
                facts = json.loads(content)
            else:
                # Sometimes model wraps in markdown
                import re

                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    facts = json.loads(match.group())
                else:
                    facts = []

            unique_facts = self._dedupe_facts(facts)

            for fact in unique_facts:
                if isinstance(fact, str) and fact.strip():
                    await self._store_fact(fact.strip(), conv_id)

            if unique_facts:
                logger.info(f"Extracted {len(unique_facts)} facts: {unique_facts}")

        except Exception as e:
            logger.error(f"Fact extraction LLM error: {e}")

    async def _store_fact(self, fact: str, conv_id: int) -> None:
        """Store a fact, updating if a similar one exists."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        fact = fact.strip()
        fact_key = self._canonicalize_fact(fact)
        if not fact_key:
            return

        now = time.time()

        try:
            cursor = await self._db.execute(
                "SELECT id FROM facts WHERE fact_key = ? LIMIT 1", (fact_key,)
            )
            row = await cursor.fetchone()

            if row:
                await self._db.execute(
                    """UPDATE facts
                       SET updated_at = ?, source_conversation_id = ?
                       WHERE fact_key = ?""",
                    (now, conv_id, fact_key),
                )
                await self._db.commit()
                return

            embedding = await self._embed(fact)
            await self._db.execute(
                """INSERT INTO facts (fact, fact_key, embedding, source_conversation_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(fact_key) DO UPDATE SET updated_at=excluded.updated_at, source_conversation_id=excluded.source_conversation_id""",
                (fact, fact_key, json.dumps(embedding), conv_id, now, now),
            )
            await self._db.commit()
        except Exception as e:
            logger.debug(f"Fact store error (may be duplicate): {e}")

    @staticmethod
    def _canonicalize_fact(fact: str) -> str:
        lowered = fact.strip().lower().replace("\u2019", "'")
        tokens = re.findall(r"[a-z0-9]+", lowered)
        return " ".join(tokens)

    def _dedupe_facts(self, facts: list) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for fact in facts:
            if not isinstance(fact, str):
                continue
            cleaned = fact.strip()
            if not cleaned:
                continue
            key = self._canonicalize_fact(cleaned)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(cleaned)
        return unique

    async def _migrate_fact_keys(self) -> None:
        if not self._db:
            return

        cursor = await self._db.execute("PRAGMA table_info(facts)")
        columns = [row[1] for row in await cursor.fetchall()]
        if "fact_key" not in columns:
            await self._db.execute("ALTER TABLE facts ADD COLUMN fact_key TEXT")

        cursor = await self._db.execute(
            "SELECT id, fact FROM facts ORDER BY updated_at DESC, id DESC"
        )
        rows = await cursor.fetchall()
        if not rows:
            await self._db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_fact_key_unique ON facts(fact_key)"
            )
            await self._db.commit()
            return

        seen_keys: dict[str, int] = {}
        delete_ids: list[int] = []
        for row_id, fact in rows:
            key = self._canonicalize_fact(fact)
            if not key:
                delete_ids.append(row_id)
                continue
            if key in seen_keys:
                delete_ids.append(row_id)
                continue
            seen_keys[key] = row_id
            await self._db.execute(
                "UPDATE facts SET fact_key = ? WHERE id = ?", (key, row_id)
            )

        if delete_ids:
            placeholders = ",".join("?" * len(delete_ids))
            await self._db.execute(
                f"DELETE FROM facts WHERE id IN ({placeholders})",
                delete_ids,
            )

        await self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_fact_key_unique ON facts(fact_key)"
        )
        await self._db.commit()

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search for relevant memories using cosine similarity."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        query_embedding = await self._embed(query)
        query_vec = np.array(query_embedding)

        results = []

        # Search conversations
        cursor = await self._db.execute(
            "SELECT id, user_message, assistant_message, embedding, timestamp FROM conversations"
        )
        rows = await cursor.fetchall()

        for row in rows:
            stored_vec = np.array(json.loads(row[3]))
            similarity = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
            )
            if similarity > config_module.config.memory.similarity_threshold:
                results.append(
                    {
                        "type": "conversation",
                        "user_message": row[1],
                        "assistant_message": row[2],
                        "similarity": similarity,
                        "timestamp": row[4],
                    }
                )

        # Search facts (these get boosted because they're distilled knowledge)
        cursor = await self._db.execute(
            "SELECT id, fact, embedding, created_at FROM facts"
        )
        fact_rows = await cursor.fetchall()

        for row in fact_rows:
            stored_vec = np.array(json.loads(row[2]))
            similarity = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
            )
            # Facts get a 0.1 boost — they're more valuable than raw conversations
            if similarity > config_module.config.memory.similarity_threshold:
                results.append(
                    {
                        "type": "fact",
                        "fact": row[1],
                        "similarity": similarity + 0.1,
                        "timestamp": row[3],
                    }
                )

        # Search actions (tool calls — what Aether *did*)
        cursor = await self._db.execute(
            "SELECT id, tool_name, arguments, output, error, embedding, timestamp FROM actions"
        )
        action_rows = await cursor.fetchall()

        for row in action_rows:
            stored_vec = np.array(json.loads(row[5]))
            similarity = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
            )
            if similarity > config_module.config.memory.similarity_threshold:
                results.append(
                    {
                        "type": "action",
                        "tool_name": row[1],
                        "arguments": row[2],
                        "output": row[3][:200],  # Preview only
                        "error": bool(row[4]),
                        "similarity": similarity
                        + 0.05,  # Slight boost over conversations
                        "timestamp": row[6],
                    }
                )

        # Search session summaries
        cursor = await self._db.execute(
            "SELECT id, summary, embedding, ended_at FROM sessions"
        )
        session_rows = await cursor.fetchall()

        for row in session_rows:
            stored_vec = np.array(json.loads(row[2]))
            similarity = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8)
            )
            if similarity > config_module.config.memory.similarity_threshold:
                results.append(
                    {
                        "type": "session",
                        "summary": row[1],
                        "similarity": similarity + 0.05,
                        "timestamp": row[3],
                    }
                )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    async def get_facts(self) -> list[str]:
        """Get all stored facts."""
        if not self._db:
            return []

        cursor = await self._db.execute(
            "SELECT fact FROM facts ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def store_preference(self, fact: str) -> None:
        """Store a notification preference as a searchable fact."""
        await self._store_fact(fact, conv_id=0)

    async def get_recent(self, limit: int = 10) -> list[dict]:
        """Get the most recent conversations."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        cursor = await self._db.execute(
            "SELECT id, user_message, assistant_message, timestamp FROM conversations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "user_message": r[1],
                "assistant_message": r[2],
                "timestamp": r[3],
            }
            for r in rows
        ]

    async def get_last_session_end(self) -> float | None:
        """Get the end timestamp of the most recent session, or None if no sessions."""
        if not self._db:
            return None

        cursor = await self._db.execute(
            "SELECT ended_at FROM sessions ORDER BY ended_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    # --- Action memory (v0.07) ---

    async def add_action(
        self,
        tool_name: str,
        arguments: dict,
        output: str,
        error: bool = False,
        session_id: str | None = None,
    ) -> None:
        """Store a tool call and its result."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        # Truncate output to keep DB manageable
        max_chars = config_module.config.memory.action_output_max_chars
        truncated_output = output[:max_chars]

        # Embed a human-readable summary for search
        args_summary = ", ".join(f"{k}={v}" for k, v in arguments.items())
        embed_text = f"Used {tool_name}({args_summary}): {truncated_output[:200]}"
        embedding = await self._embed(embed_text)

        await self._db.execute(
            """INSERT INTO actions
               (tool_name, arguments, output, error, embedding, session_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                tool_name,
                json.dumps(arguments),
                truncated_output,
                error,
                json.dumps(embedding),
                session_id,
                time.time(),
            ),
        )
        await self._db.commit()
        logger.info(f"Stored action: {tool_name}({args_summary[:60]})")

    # --- Session summaries (v0.07) ---

    async def add_session_summary(
        self,
        session_id: str,
        summary: str,
        started_at: float,
        ended_at: float,
        turns: int,
        tools_used: list[str] | None = None,
    ) -> None:
        """Store a session summary for cross-session continuity."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        embedding = await self._embed(summary)

        await self._db.execute(
            """INSERT INTO sessions
               (session_id, summary, embedding, started_at, ended_at, turns, tools_used)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                summary,
                json.dumps(embedding),
                started_at,
                ended_at,
                turns,
                json.dumps(tools_used or []),
            ),
        )
        await self._db.commit()
        logger.info(f"Stored session summary: {summary[:80]}...")

    async def get_session_summaries(self, limit: int | None = None) -> list[dict]:
        """Get the most recent session summaries."""
        if not self._db:
            return []

        limit = limit or config_module.config.memory.session_summary_limit
        cursor = await self._db.execute(
            "SELECT session_id, summary, started_at, ended_at, turns, tools_used "
            "FROM sessions ORDER BY ended_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "session_id": r[0],
                "summary": r[1],
                "started_at": r[2],
                "ended_at": r[3],
                "turns": r[4],
                "tools_used": json.loads(r[5]) if r[5] else [],
            }
            for r in rows
        ]

    # --- Action compaction (v0.07) ---

    async def _compact_old_actions(self) -> None:
        """Summarize old actions into facts and delete the raw rows.

        Runs on startup. Actions older than action_retention_days get
        summarized by the LLM and stored as facts.
        """
        if not self._db:
            return

        cutoff = time.time() - (
            config_module.config.memory.action_retention_days * 86400
        )

        cursor = await self._db.execute(
            "SELECT id, tool_name, arguments, output, error, timestamp FROM actions WHERE timestamp < ?",
            (cutoff,),
        )
        old_actions = list(await cursor.fetchall())

        if not old_actions:
            return

        logger.info(f"Compacting {len(old_actions)} old actions into facts")

        # Build a summary of old actions for the LLM
        action_lines = []
        action_ids = []
        for row in old_actions:
            action_ids.append(row[0])
            args = json.loads(row[2])
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            status = "failed" if row[4] else "succeeded"
            action_lines.append(f"- {row[1]}({args_str}) → {status}: {row[3][:100]}")

        if not action_lines:
            return

        actions_text = "\n".join(action_lines[:50])  # Cap at 50 for prompt size

        try:
            response = await self._get_openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Summarize these past tool actions into 2-5 key facts about what was accomplished. "
                            "Focus on what was created, modified, or discovered. "
                            "Return a JSON array of fact strings.\n\n"
                            f"Actions:\n{actions_text}\n\nFacts (JSON array):"
                        ),
                    }
                ],
                max_tokens=300,
                temperature=0.0,
            )

            content = response.choices[0].message.content
            if content:
                content = content.strip()
                if content.startswith("["):
                    facts = json.loads(content)
                else:
                    import re

                    match = re.search(r"\[.*\]", content, re.DOTALL)
                    facts = json.loads(match.group()) if match else []

                for fact in facts:
                    if isinstance(fact, str) and fact.strip():
                        await self._store_fact(fact.strip(), 0)

                logger.info(
                    f"Compacted {len(old_actions)} actions into {len(facts)} facts"
                )

            # Delete the old action rows
            placeholders = ",".join("?" * len(action_ids))
            await self._db.execute(
                f"DELETE FROM actions WHERE id IN ({placeholders})",
                action_ids,
            )
            await self._db.commit()

        except Exception as e:
            logger.error(f"Action compaction failed: {e}")

    async def _embed(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        client = self._get_openai_client()
        model = config_module.config.memory.embedding_model
        llm_cfg = config_module.config.llm
        if (
            llm_cfg.base_url
            and "openrouter" in llm_cfg.base_url.lower()
            and "/" not in model
        ):
            # Embeddings on OpenRouter should use an embedding-capable provider.
            # Default to OpenAI embedding models unless caller already provided a scoped model.
            model = f"openai/{model}"

        response = await client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding

    def _get_openai_client(self) -> AsyncOpenAI:
        """Return an OpenAI client configured for the current runtime config."""
        current_base_url = config_module.config.llm.base_url or ""
        current_api_key = config_module.config.llm.api_key or ""

        # Test hook: if a client was injected directly, prefer it.
        if (
            self.openai is not None
            and self._openai_base_url is None
            and self._openai_api_key is None
        ):
            return self.openai

        if (
            self.openai
            and self._openai_base_url == current_base_url
            and self._openai_api_key == current_api_key
        ):
            return self.openai

        if current_base_url:
            self.openai = AsyncOpenAI(
                api_key=current_api_key,
                base_url=current_base_url,
            )
        else:
            self.openai = AsyncOpenAI(api_key=current_api_key)
        self._openai_base_url = current_base_url
        self._openai_api_key = current_api_key
        return self.openai
