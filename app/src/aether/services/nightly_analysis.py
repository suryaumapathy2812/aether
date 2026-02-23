"""
Nightly Analysis Service — periodic memory maintenance and proactive insights.

Runs as a background E Worker task on a 24-hour timer (Requirements.md §6.2).
Performs four analysis functions:

1. Decision extraction from patterns — find behavioral patterns in recent memories
2. Proactive opportunity detection — identify time-sensitive opportunities
3. Fact consolidation — merge duplicates, flag stale facts
4. Memory health metrics — count items per bucket, log metrics

All LLM calls go through LLMCore for consistent contracts.
Must complete within 60 seconds (Requirements.md §6.6).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aether.llm.contracts import LLMEventType, LLMRequestEnvelope

if TYPE_CHECKING:
    from aether.llm.core import LLMCore
    from aether.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# ─── Prompts ─────────────────────────────────────────────────────

DECISION_EXTRACTION_PROMPT = """You are Aether's nightly analysis engine. Given recent memories and existing decisions, identify behavioral patterns that should become new decisions (rules about how the agent should behave).

RECENT MEMORIES (last 7 days):
{memories}

EXISTING DECISIONS:
{decisions}

Rules:
- Only propose decisions that are NOT already covered by existing decisions
- Decisions are behavioral rules: "Don't notify about X after Y", "User prefers Z format"
- Each decision should be actionable and specific
- Skip if no clear patterns emerge

Return ONLY a JSON array of decision strings. If none, return [].
Example: ["User prefers code examples over explanations", "Don't send calendar reminders before 8am"]"""

PROACTIVE_OPPORTUNITY_PROMPT = """You are Aether's proactive opportunity detector. Given the user's facts and recent memories, identify time-sensitive opportunities or reminders the user should know about.

FACTS:
{facts}

RECENT MEMORIES:
{memories}

Rules:
- Only flag genuinely time-sensitive or high-value opportunities
- Include a suggested delivery type: "nudge" (low priority), "surface" (normal), "interrupt" (urgent)
- Include a suggested delivery time description (e.g. "morning", "immediately", "before 5pm")
- Skip if nothing actionable

Return ONLY a JSON array of objects. If none, return [].
Example: [{{"text": "Q3 deadline is tomorrow — review status?", "delivery_type": "surface", "deliver_at": "morning"}}]"""

FACT_CONSOLIDATION_PROMPT = """You are Aether's memory maintenance engine. Review these facts for quality issues.

FACTS:
{facts}

Identify:
1. Duplicate or near-duplicate facts (suggest merging into one)
2. Contradictory facts (flag both)
3. Likely stale facts (outdated information)

Return ONLY a JSON object:
{{
  "merge": [{{"keep": "fact text to keep", "remove": ["fact text to remove", ...]}}],
  "stale": ["fact text that seems outdated", ...]
}}

If no issues found, return {{"merge": [], "stale": []}}"""


@dataclass
class NightlyAnalysisResult:
    """Summary of nightly analysis run."""

    new_decisions: int = 0
    candidate_notifications: list[dict] = field(default_factory=list)
    facts_consolidated: int = 0
    facts_flagged_stale: int = 0
    memory_health: dict = field(default_factory=dict)
    duration_ms: int = 0


class NightlyAnalysisService:
    """
    Nightly memory analysis and proactive insight generation.

    Self-contained service: takes LLMCore and MemoryStore, performs
    analysis, returns results. Called by AgentCore on a 24-hour timer.
    """

    def __init__(
        self,
        llm_core: "LLMCore",
        memory_store: "MemoryStore",
    ) -> None:
        self._llm_core = llm_core
        self._memory_store = memory_store

    async def run_analysis(self) -> NightlyAnalysisResult:
        """Run all four analysis functions. Returns a summary.

        Must complete within 60 seconds (Requirements.md §6.6).
        Each function is independent — failures in one don't block others.
        """
        started = time.time()
        result = NightlyAnalysisResult()

        # 1. Decision extraction from patterns
        try:
            result.new_decisions = await self._extract_decisions()
        except Exception as e:
            logger.error("Nightly: decision extraction failed: %s", e, exc_info=True)

        # 2. Proactive opportunity detection
        try:
            result.candidate_notifications = await self._detect_opportunities()
        except Exception as e:
            logger.error("Nightly: opportunity detection failed: %s", e, exc_info=True)

        # 3. Fact consolidation
        try:
            consolidated, stale = await self._consolidate_facts()
            result.facts_consolidated = consolidated
            result.facts_flagged_stale = stale
        except Exception as e:
            logger.error("Nightly: fact consolidation failed: %s", e, exc_info=True)

        # 4. Memory health metrics (no LLM call)
        try:
            result.memory_health = await self._collect_health_metrics()
        except Exception as e:
            logger.error("Nightly: health metrics failed: %s", e, exc_info=True)

        result.duration_ms = round((time.time() - started) * 1000)

        logger.info(
            "Nightly analysis complete in %dms: decisions=%d, notifications=%d, "
            "consolidated=%d, stale=%d, health=%s",
            result.duration_ms,
            result.new_decisions,
            len(result.candidate_notifications),
            result.facts_consolidated,
            result.facts_flagged_stale,
            result.memory_health,
        )

        return result

    # ─── Analysis Functions ──────────────────────────────────────

    async def _extract_decisions(self) -> int:
        """Load recent memories + existing decisions, ask LLM for new decisions.

        Returns the number of new decisions stored.
        """
        # Load recent memories (last 7 days)
        seven_days_ago = time.time() - (7 * 24 * 3600)
        all_memories = await self._memory_store.get_memories(limit=100)
        recent_memories = [
            m for m in all_memories if m.get("created_at", 0) >= seven_days_ago
        ]

        if not recent_memories:
            logger.debug("Nightly: no recent memories for decision extraction")
            return 0

        # Load existing decisions
        existing_decisions = await self._memory_store.get_decisions(active_only=True)

        # Format for prompt
        memories_text = "\n".join(
            f"- [{m.get('category', 'unknown')}] {m['memory']}"
            for m in recent_memories[:50]  # Cap to keep prompt manageable
        )
        decisions_text = (
            "\n".join(f"- {d['decision']}" for d in existing_decisions[:30])
            or "(none yet)"
        )

        prompt = DECISION_EXTRACTION_PROMPT.format(
            memories=memories_text,
            decisions=decisions_text,
        )

        envelope = LLMRequestEnvelope(
            kind="nightly_decision_extract",
            modality="system",
            messages=[{"role": "user", "content": prompt}],
            policy={"max_tokens": 500, "temperature": 0.3},
        )

        response_text = await self._collect_response(envelope)
        new_decisions = self._parse_json_array(response_text)

        count = 0
        for decision in new_decisions:
            if isinstance(decision, str) and decision.strip():
                try:
                    await self._memory_store.store_decision(
                        decision.strip(),
                        category="behavior",
                        source="nightly_analysis",
                    )
                    count += 1
                except Exception as e:
                    logger.debug("Failed to store decision: %s", e)

        if count:
            logger.info("Nightly: extracted %d new decisions", count)
        return count

    async def _detect_opportunities(self) -> list[dict]:
        """Load facts + recent memories, ask LLM for proactive opportunities.

        Returns a list of candidate notifications.
        """
        facts = await self._memory_store.get_facts()
        all_memories = await self._memory_store.get_memories(limit=50)

        if not facts and not all_memories:
            logger.debug("Nightly: no facts or memories for opportunity detection")
            return []

        facts_text = "\n".join(f"- {f}" for f in facts[:30]) or "(none)"
        memories_text = (
            "\n".join(
                f"- [{m.get('category', 'unknown')}] {m['memory']}"
                for m in all_memories[:30]
            )
            or "(none)"
        )

        prompt = PROACTIVE_OPPORTUNITY_PROMPT.format(
            facts=facts_text,
            memories=memories_text,
        )

        envelope = LLMRequestEnvelope(
            kind="nightly_proactive_check",
            modality="system",
            messages=[{"role": "user", "content": prompt}],
            policy={"max_tokens": 500, "temperature": 0.3},
        )

        response_text = await self._collect_response(envelope)
        candidates = self._parse_json_array(response_text)

        # Normalize candidates into dicts with required fields
        notifications: list[dict] = []
        for item in candidates:
            if isinstance(item, dict) and item.get("text"):
                notifications.append(
                    {
                        "text": str(item["text"]).strip(),
                        "delivery_type": str(
                            item.get("delivery_type", "surface")
                        ).strip(),
                        "deliver_at": str(item.get("deliver_at", "")).strip() or None,
                    }
                )
            elif isinstance(item, str) and item.strip():
                # Fallback: plain string → surface notification
                notifications.append(
                    {
                        "text": item.strip(),
                        "delivery_type": "surface",
                        "deliver_at": None,
                    }
                )

        if notifications:
            logger.info(
                "Nightly: detected %d proactive opportunities", len(notifications)
            )
        return notifications

    async def _consolidate_facts(self) -> tuple[int, int]:
        """Load all facts, ask LLM for consolidation suggestions, apply them.

        Returns (facts_consolidated, facts_flagged_stale).
        """
        facts = await self._memory_store.get_facts()

        if len(facts) < 3:
            logger.debug("Nightly: too few facts (%d) for consolidation", len(facts))
            return 0, 0

        facts_text = "\n".join(f"- {f}" for f in facts[:80])  # Cap for prompt size

        prompt = FACT_CONSOLIDATION_PROMPT.format(facts=facts_text)

        envelope = LLMRequestEnvelope(
            kind="nightly_fact_consolidation",
            modality="system",
            messages=[{"role": "user", "content": prompt}],
            policy={"max_tokens": 1000, "temperature": 0.3},
        )

        response_text = await self._collect_response(envelope)
        consolidation = self._parse_json_object(response_text)

        merged_count = 0
        stale_count = 0

        # Apply merges: remove duplicate facts
        for merge_item in consolidation.get("merge", []):
            if not isinstance(merge_item, dict):
                continue
            remove_list = merge_item.get("remove", [])
            for fact_text in remove_list:
                if isinstance(fact_text, str) and fact_text.strip():
                    try:
                        await self._remove_fact_by_text(fact_text.strip())
                        merged_count += 1
                    except Exception as e:
                        logger.debug("Failed to remove merged fact: %s", e)

        # Flag stale facts by reducing confidence
        for stale_fact in consolidation.get("stale", []):
            if isinstance(stale_fact, str) and stale_fact.strip():
                try:
                    await self._reduce_fact_confidence(stale_fact.strip())
                    stale_count += 1
                except Exception as e:
                    logger.debug("Failed to flag stale fact: %s", e)

        if merged_count or stale_count:
            logger.info(
                "Nightly: consolidated %d facts, flagged %d stale",
                merged_count,
                stale_count,
            )
        return merged_count, stale_count

    async def _collect_health_metrics(self) -> dict:
        """Count items per memory bucket. No LLM call needed."""
        facts = await self._memory_store.get_facts()
        memories = await self._memory_store.get_memories(limit=10000)
        decisions = await self._memory_store.get_decisions(active_only=False)

        health = {
            "facts_count": len(facts),
            "memories_count": len(memories),
            "decisions_count": len(decisions),
        }

        logger.info(
            "Memory health: facts=%d, memories=%d, decisions=%d",
            health["facts_count"],
            health["memories_count"],
            health["decisions_count"],
        )
        return health

    # ─── Helpers ─────────────────────────────────────────────────

    async def _collect_response(self, envelope: LLMRequestEnvelope) -> str:
        """Collect full text response from LLMCore."""
        chunks: list[str] = []

        async for event in self._llm_core.generate_with_tools(envelope):
            if event.event_type == LLMEventType.TEXT_CHUNK.value:
                chunks.append(event.payload.get("text", ""))

        return " ".join(chunks)

    def _parse_json_array(self, response_text: str) -> list:
        """Parse a JSON array from LLM response, with fallback."""
        response_text = response_text.strip()

        try:
            if response_text.startswith("["):
                return json.loads(response_text)
            # Model may wrap in markdown code block
            match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug("Could not parse JSON array: %s — %s", e, response_text[:100])

        return []

    def _parse_json_object(self, response_text: str) -> dict:
        """Parse a JSON object from LLM response, with fallback."""
        response_text = response_text.strip()

        try:
            if response_text.startswith("{"):
                return json.loads(response_text)
            # Model may wrap in markdown code block
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug("Could not parse JSON object: %s — %s", e, response_text[:100])

        return {"merge": [], "stale": []}

    async def _remove_fact_by_text(self, fact_text: str) -> None:
        """Remove a fact by matching its text (best-effort)."""
        db = self._memory_store._db
        if not db:
            return

        # Use canonicalized key for matching (same as MemoryStore._canonicalize_fact)
        fact_key = self._memory_store._canonicalize_fact(fact_text)
        if not fact_key:
            return

        await db.execute("DELETE FROM facts WHERE fact_key = ?", (fact_key,))
        await db.commit()

    async def _reduce_fact_confidence(self, fact_text: str) -> None:
        """Reduce a fact's updated_at to deprioritize it (stale signal).

        Since the facts table doesn't have a confidence column, we push
        updated_at back by 30 days to lower its ranking in recency-sorted queries.
        """
        db = self._memory_store._db
        if not db:
            return

        fact_key = self._memory_store._canonicalize_fact(fact_text)
        if not fact_key:
            return

        # Push updated_at back by 30 days as a staleness signal
        stale_time = time.time() - (30 * 24 * 3600)
        await db.execute(
            "UPDATE facts SET updated_at = ? WHERE fact_key = ?",
            (stale_time, fact_key),
        )
        await db.commit()
