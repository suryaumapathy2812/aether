"""
Aether Memory Store — SQLite + OpenAI embeddings.

Stores conversation turns with vector embeddings for semantic search.
v0.01: Flat table, brute-force cosine similarity. Simple and correct.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import aiosqlite
import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
DB_PATH = Path(os.getenv("AETHER_DB_PATH", "aether_memory.db"))


class MemoryStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
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
        await self._db.commit()
        logger.info(f"Memory store initialized at {self.db_path}")

    async def stop(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def add(self, user_message: str, assistant_message: str) -> None:
        """Store a conversation turn with its embedding."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        # Embed the combined conversation for richer semantic matching
        combined = f"User: {user_message}\nAssistant: {assistant_message}"
        embedding = await self._embed(combined)

        import time
        await self._db.execute(
            "INSERT INTO conversations (user_message, assistant_message, embedding, timestamp) VALUES (?, ?, ?, ?)",
            (user_message, assistant_message, json.dumps(embedding), time.time()),
        )
        await self._db.commit()
        logger.debug(f"Stored memory: {user_message[:50]}...")

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search for relevant memories using cosine similarity."""
        if not self._db:
            raise RuntimeError("Memory store not started")

        query_embedding = await self._embed(query)

        # Fetch all embeddings and compute similarity (brute force — fine for <10K rows)
        cursor = await self._db.execute(
            "SELECT id, user_message, assistant_message, embedding, timestamp FROM conversations"
        )
        rows = await cursor.fetchall()

        if not rows:
            return []

        # Score each row
        scored = []
        query_vec = np.array(query_embedding)
        for row in rows:
            stored_vec = np.array(json.loads(row[3]))
            similarity = float(np.dot(query_vec, stored_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8
            ))
            scored.append({
                "id": row[0],
                "user_message": row[1],
                "assistant_message": row[2],
                "similarity": similarity,
                "timestamp": row[4],
            })

        # Sort by similarity descending, return top N
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:limit]

    async def _embed(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        response = await self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

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
