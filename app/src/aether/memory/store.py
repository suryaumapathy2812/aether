"""
Aether Memory Store — SQLite + OpenAI embeddings.

v0.03: Two-tier memory:
1. conversations — raw exchanges (user said X, assistant said Y)
2. facts — extracted knowledge ("user's name is Surya", "user prefers Python")

Facts are extracted async after each conversation turn using a lightweight
LLM call. This is what makes Aether feel like it *knows* you.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import aiosqlite
import numpy as np
from openai import AsyncOpenAI

from aether.core.config import config

logger = logging.getLogger(__name__)

FACT_EXTRACTION_PROMPT = """Extract key facts from this conversation turn. Focus on:
- Personal details (name, location, job, preferences)
- Preferences and opinions ("I like...", "I prefer...", "I hate...")
- Plans and goals ("I'm working on...", "I want to...")
- Relationships ("my wife...", "my friend...")
- Any specific factual information the user shared

Return a JSON array of fact strings. Each fact should be a short, standalone statement.
If no significant facts are present, return an empty array [].

Example output: ["User's name is Alex", "User works at Google", "User prefers dark mode"]

Conversation:
User: {user_message}
Assistant: {assistant_message}

Facts (JSON array):"""


class MemoryStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or Path(config.memory.db_path)
        self.openai = AsyncOpenAI()
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

        # New: facts table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL UNIQUE,
                embedding TEXT NOT NULL,
                source_conversation_id INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        await self._db.commit()
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
        conv_id = cursor.lastrowid
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
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap for extraction
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )

            content = response.choices[0].message.content.strip()

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

            for fact in facts:
                if isinstance(fact, str) and fact.strip():
                    await self._store_fact(fact.strip(), conv_id)

            if facts:
                logger.info(f"Extracted {len(facts)} facts: {facts}")

        except Exception as e:
            logger.error(f"Fact extraction LLM error: {e}")

    async def _store_fact(self, fact: str, conv_id: int) -> None:
        """Store a fact, updating if a similar one exists."""
        embedding = await self._embed(fact)
        now = time.time()

        try:
            await self._db.execute(
                """INSERT INTO facts (fact, embedding, source_conversation_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(fact) DO UPDATE SET updated_at=?""",
                (fact, json.dumps(embedding), conv_id, now, now, now),
            )
            await self._db.commit()
        except Exception as e:
            logger.debug(f"Fact store error (may be duplicate): {e}")

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
            if similarity > config.memory.similarity_threshold:
                results.append({
                    "type": "conversation",
                    "user_message": row[1],
                    "assistant_message": row[2],
                    "similarity": similarity,
                    "timestamp": row[4],
                })

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
            if similarity > config.memory.similarity_threshold:
                results.append({
                    "type": "fact",
                    "fact": row[1],
                    "similarity": similarity + 0.1,
                    "timestamp": row[3],
                })

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
            {"id": r[0], "user_message": r[1], "assistant_message": r[2], "timestamp": r[3]}
            for r in rows
        ]

    async def _embed(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        response = await self.openai.embeddings.create(
            model=config.memory.embedding_model,
            input=text,
        )
        return response.data[0].embedding
