"""
Kernel Interface — abstract interface for job orchestration.

The kernel provides a unified interface for:
- Submitting jobs (submit)
- Streaming events (stream)
- Awaiting results (await_result)
- Canceling jobs (cancel)

Implementations:
- KernelScheduler: Full scheduler with priority queues (future)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from aether.kernel.contracts import KernelEvent, KernelRequest, KernelResult


class KernelInterface(ABC):
    """
    Abstract interface for the kernel.

    All methods are async and return/emit kernel contracts.
    Implementations handle routing, scheduling, and execution.
    """

    @abstractmethod
    async def submit(self, request: KernelRequest) -> str:
        """
        Submit a job to the kernel.

        Args:
            request: The job request with kind, modality, payload, etc.

        Returns:
            job_id: Unique identifier for tracking this job

        The job is queued and will be processed according to priority.
        Use stream() or await_result() to get the output.
        """
        ...

    @abstractmethod
    async def stream(self, job_id: str) -> AsyncGenerator[KernelEvent, None]:
        """
        Stream events from a running job.

        Yields events in order (sequence is monotonic):
        - text_chunk: Partial text output
        - tool_result: Tool execution result
        - status: Status update (thinking, working, etc.)
        - error: Error occurred
        - stream_end: Job completed

        Args:
            job_id: The job to stream

        Yields:
            KernelEvent: Events from the job execution
        """
        ...
        yield KernelEvent(job_id="", stream_type="", payload={}, sequence=0)  # type: ignore

    @abstractmethod
    async def await_result(
        self, job_id: str, timeout_ms: int | None = None
    ) -> KernelResult:
        """
        Wait for job completion and return the result.

        Args:
            job_id: The job to await
            timeout_ms: Optional timeout in milliseconds

        Returns:
            KernelResult: Terminal result with status, payload, metrics

        Raises:
            TimeoutError: If timeout_ms is exceeded
        """
        ...

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: The job to cancel

        Returns:
            True if the job was canceled
            False if the job was already complete or not found
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict:
        """
        Check kernel health.

        Returns:
            Dict with status, queue depths, worker counts, etc.
        """
        ...
