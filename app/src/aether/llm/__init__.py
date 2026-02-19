"""
LLM Package — single entry point for all LLM calls.

This package provides:
- LLMRequestEnvelope: Fixed input structure for ALL LLM calls
- LLMEventEnvelope: Streamed events from LLM
- LLMResultEnvelope: Terminal result
- LLMCore: Single entry point with tool calling loop
- ContextBuilder: Builds envelopes with skill/plugin injection

The LLM is treated as shared compute, not a session-bound pipeline.
All consumers (ReplyService, MemoryService, NotificationService) use the same interface.
"""

from aether.llm.contracts import (
    LLMRequestEnvelope,
    LLMEventEnvelope,
    LLMResultEnvelope,
    LLMEventType,
    ToolCallRequest,
    ToolResult,
)

__all__ = [
    # Contracts
    "LLMRequestEnvelope",
    "LLMEventEnvelope",
    "LLMResultEnvelope",
    "LLMEventType",
    "ToolCallRequest",
    "ToolResult",
]
