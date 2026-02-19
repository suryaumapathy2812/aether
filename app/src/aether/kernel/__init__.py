"""
Kernel Package — job orchestration layer.

The kernel provides a unified interface for submitting, streaming, and awaiting
jobs of various kinds (reply_text, reply_voice, memory_*, notification_*, etc.).

Architecture:
  AgentCore (facade) → KernelScheduler (2P+2E workers) → ServiceRouter → Services
"""

from aether.kernel.contracts import (
    JobKind,
    JobModality,
    JobPriority,
    JobStatus,
    KernelEvent,
    KernelRequest,
    KernelResult,
)
from aether.kernel.scheduler import KernelScheduler, ServiceRouter

__all__ = [
    # Contracts
    "KernelRequest",
    "KernelEvent",
    "KernelResult",
    "JobKind",
    "JobModality",
    "JobPriority",
    "JobStatus",
    # Scheduler
    "KernelScheduler",
    "ServiceRouter",
]
