"""
System Adapter — handles internal/background jobs with no user I/O.

Used for:
- Memory fact extraction
- Session summaries
- Action compaction
- Notification classification
- Any fire-and-forget background work

No STT/TTS. No user-facing output. Results are stored or forwarded
to the appropriate service.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator

from aether.kernel.contracts import JobModality
from aether.modality.base import ModalityAdapter

if TYPE_CHECKING:
    from aether.transport.core_msg import CoreMsg

logger = logging.getLogger(__name__)


class SystemAdapter(ModalityAdapter):
    """
    System modality adapter — no user I/O.

    Handles background/internal jobs that don't produce user-facing output.
    Results are consumed by services (MemoryService, NotificationService)
    rather than sent to a transport.
    """

    @property
    def modality(self) -> str:
        return JobModality.SYSTEM.value

    async def handle_input(
        self,
        msg: "CoreMsg",
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """
        System adapter doesn't process transport input directly.

        Background jobs are submitted through the kernel, not through
        transport messages. This is a no-op that logs a warning.
        """
        logger.warning(
            "SystemAdapter.handle_input called — system jobs should be "
            "submitted through the kernel, not transport messages."
        )
        return
        yield  # type: ignore[misc]

    async def handle_output(
        self,
        event_type: str,
        payload: dict[str, Any],
        session_state: dict[str, Any],
    ) -> AsyncGenerator["CoreMsg", None]:
        """
        System adapter doesn't produce transport output.

        Background job results are consumed by services directly.
        This is a no-op.
        """
        logger.debug(
            f"SystemAdapter.handle_output called with {event_type} — "
            "system results are consumed by services, not transports."
        )
        return
        yield  # type: ignore[misc]
