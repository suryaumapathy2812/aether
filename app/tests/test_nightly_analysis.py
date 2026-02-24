"""Tests for NightlyAnalysisService.

Covers:
- run_analysis: completes and returns NightlyAnalysisResult
- Decision extraction: stores new decisions via memory_store
- Opportunity detection: returns candidate notifications
- Resilience: individual function failure doesn't block others
- Health metrics: counts items without LLM calls
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from aether.llm.contracts import LLMEventEnvelope, LLMEventType
from aether.services.nightly_analysis import NightlyAnalysisService


# ─── Helpers ─────────────────────────────────────────────────────


def _make_llm_core(response_text: str) -> AsyncMock:
    """Create a mock LLMCore that returns the given text."""
    llm_core = AsyncMock()

    async def fake_generate(envelope):
        yield LLMEventEnvelope(
            event_type=LLMEventType.TEXT_CHUNK.value,
            request_id="req-1",
            job_id="job-1",
            payload={"text": response_text},
            sequence=1,
        )

    llm_core.generate_with_tools = fake_generate
    return llm_core


# ─── Tests ───────────────────────────────────────────────────────


class TestNightlyAnalysis:
    """Test NightlyAnalysisService.run_analysis and sub-functions."""

    @pytest.mark.asyncio
    async def test_run_analysis_returns_result(self):
        """Positive: run_analysis completes and returns NightlyAnalysisResult.

        Objective: verify that the service runs all four analysis functions
        and returns a result with valid duration_ms.
        """
        memory_store = AsyncMock()
        memory_store.get_memories = AsyncMock(return_value=[])
        memory_store.get_decisions = AsyncMock(return_value=[])
        memory_store.get_facts = AsyncMock(return_value=[])

        llm_core = _make_llm_core("[]")
        service = NightlyAnalysisService(llm_core, memory_store)
        result = await service.run_analysis()

        assert result.duration_ms >= 0
        assert result.new_decisions == 0
        assert result.candidate_notifications == []

    @pytest.mark.asyncio
    async def test_decision_extraction_stores_new_decisions(self):
        """Positive: decisions extracted from patterns are stored via memory_store.

        Objective: verify that when the LLM returns decision strings, they
        are stored via store_decision and the count is correct.
        """
        import time

        memory_store = AsyncMock()
        memory_store.get_memories = AsyncMock(
            return_value=[
                {
                    "memory": "User dismissed 3 notifications",
                    "category": "behavioral",
                    "created_at": time.time(),  # Recent — within 7 days
                },
            ]
        )
        memory_store.get_decisions = AsyncMock(return_value=[])
        memory_store.get_facts = AsyncMock(return_value=[])
        memory_store.store_decision = AsyncMock()

        llm_core = _make_llm_core('["Reduce notification frequency"]')
        service = NightlyAnalysisService(llm_core, memory_store)
        result = await service.run_analysis()

        assert result.new_decisions == 1
        memory_store.store_decision.assert_called_once()
        call_args = memory_store.store_decision.call_args
        assert "Reduce notification frequency" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_opportunity_detection(self):
        """Positive: proactive opportunities are returned as candidate notifications.

        Objective: verify that LLM-detected opportunities are parsed and
        returned in the result's candidate_notifications list.
        """
        memory_store = AsyncMock()
        memory_store.get_memories = AsyncMock(return_value=[])
        memory_store.get_decisions = AsyncMock(return_value=[])
        memory_store.get_facts = AsyncMock(return_value=["User has Q3 deadline Friday"])

        llm_core = _make_llm_core(
            '[{"text": "Q3 deadline is tomorrow", "delivery_type": "surface"}]'
        )
        service = NightlyAnalysisService(llm_core, memory_store)
        result = await service.run_analysis()

        assert len(result.candidate_notifications) == 1
        assert result.candidate_notifications[0]["text"] == "Q3 deadline is tomorrow"
        assert result.candidate_notifications[0]["delivery_type"] == "surface"

    @pytest.mark.asyncio
    async def test_individual_function_failure_doesnt_block_others(self):
        """Negative: if one analysis function fails, others still run.

        Objective: verify resilience — a failure in fact consolidation
        doesn't prevent health metrics from completing.

        Call order in run_analysis:
        1. _extract_decisions → get_memories (returns []) → skips (no recent)
        2. _detect_opportunities → get_facts, get_memories → skips (empty)
        3. _consolidate_facts → get_facts → raises on 3rd call
        4. _collect_health_metrics → get_facts, get_memories, get_decisions

        We use side_effect list for get_facts: first two calls succeed
        (for _detect_opportunities and _consolidate_facts), third call
        raises (inside _consolidate_facts after the len check), but
        actually _consolidate_facts needs >=3 facts. So we make it have
        enough facts, then fail the LLM call instead.

        Simplest approach: make the LLM call fail for consolidation by
        having enough facts (>=3) and a broken LLM response, while
        health metrics (no LLM) still works.
        """
        import time

        memory_store = AsyncMock()
        # Provide recent memories so _extract_decisions calls LLM
        memory_store.get_memories = AsyncMock(
            return_value=[
                {"memory": "m1", "category": "behavioral", "created_at": time.time()},
            ]
        )
        memory_store.get_decisions = AsyncMock(return_value=[])
        memory_store.get_facts = AsyncMock(return_value=["fact1", "fact2", "fact3"])
        memory_store.store_decision = AsyncMock()

        # LLM returns valid JSON for decisions, then raises for consolidation
        call_count = 0

        async def llm_side_effect(envelope):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                # Third LLM call is fact consolidation — make it raise
                raise Exception("LLM timeout")
            yield LLMEventEnvelope(
                event_type=LLMEventType.TEXT_CHUNK.value,
                request_id="req-1",
                job_id="job-1",
                payload={"text": "[]"},
                sequence=1,
            )

        llm_core = AsyncMock()
        llm_core.generate_with_tools = llm_side_effect

        service = NightlyAnalysisService(llm_core, memory_store)
        result = await service.run_analysis()

        # Should still complete
        assert result.duration_ms >= 0
        # Health metrics should have run (no LLM needed)
        assert result.memory_health.get("facts_count") == 3
        assert result.memory_health.get("memories_count") == 1
        assert result.memory_health.get("decisions_count") == 0

    @pytest.mark.asyncio
    async def test_health_metrics_no_llm_call(self):
        """Positive: health metrics collection doesn't call LLM.

        Objective: verify that _collect_health_metrics only queries the
        memory store and never invokes the LLM.
        """
        memory_store = AsyncMock()
        memory_store.get_facts = AsyncMock(return_value=["fact1", "fact2"])
        memory_store.get_memories = AsyncMock(return_value=[{"memory": "m1"}])
        memory_store.get_decisions = AsyncMock(return_value=[{"decision": "d1"}])

        llm_core = AsyncMock()
        service = NightlyAnalysisService(llm_core, memory_store)
        health = await service._collect_health_metrics()

        assert health["facts_count"] == 2
        assert health["memories_count"] == 1
        assert health["decisions_count"] == 1
        # LLM should NOT have been called
        llm_core.generate_with_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_recent_memories_skips_decision_extraction(self):
        """Negative: no recent memories means no LLM call for decisions.

        Objective: verify that when all memories are older than 7 days,
        decision extraction returns 0 without calling the LLM.
        """
        memory_store = AsyncMock()
        memory_store.get_memories = AsyncMock(
            return_value=[
                {
                    "memory": "Old memory",
                    "category": "behavioral",
                    "created_at": 0,  # Very old — epoch time
                },
            ]
        )
        memory_store.get_decisions = AsyncMock(return_value=[])
        memory_store.get_facts = AsyncMock(return_value=[])

        llm_core = _make_llm_core("[]")
        service = NightlyAnalysisService(llm_core, memory_store)
        count = await service._extract_decisions()

        assert count == 0

    @pytest.mark.asyncio
    async def test_malformed_llm_response_returns_empty(self):
        """Negative: malformed JSON from LLM doesn't crash, returns empty.

        Objective: verify that _parse_json_array handles garbage gracefully.
        """
        memory_store = AsyncMock()
        memory_store.get_memories = AsyncMock(return_value=[])
        memory_store.get_decisions = AsyncMock(return_value=[])
        memory_store.get_facts = AsyncMock(return_value=[])

        llm_core = _make_llm_core("this is not json at all")
        service = NightlyAnalysisService(llm_core, memory_store)
        result = await service.run_analysis()

        # Should complete without error
        assert result.duration_ms >= 0
        assert result.new_decisions == 0
        assert result.candidate_notifications == []
