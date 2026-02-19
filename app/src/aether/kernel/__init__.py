"""
Kernel Package — job orchestration layer.

The kernel provides a unified interface for submitting, streaming, and awaiting
jobs of various kinds (reply_text, reply_voice, memory_*, notification_*, etc.).

This is the foundation for the "LLM-kernel OS" architecture where LLM is treated
as shared compute, not a session-bound pipeline.

KernelCore is the primary CoreInterface implementation — the single entry point
for all transport → core processing.
"""

from aether.kernel.contracts import (
    KernelRequest,
    KernelEvent,
    KernelResult,
    JobKind,
    JobModality,
    JobPriority,
    JobStatus,
)
from aether.kernel.core import KernelCore
from aether.kernel.interface import KernelInterface
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
    # Interface
    "KernelInterface",
    # Core implementation
    "KernelCore",
    # Scheduler
    "KernelScheduler",
    "ServiceRouter",
]
