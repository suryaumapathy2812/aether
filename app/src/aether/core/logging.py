"""
Aether Logging — clean, colorized, pipeline-aware logging.

Features:
- Color formatter for dev mode (auto-detects TTY)
- Suppresses noisy third-party loggers (httpx, httpcore, openai)
- Configurable via AETHER_LOG_LEVEL and AETHER_LOG_COLOR
- Pipeline timing helper for STT -> LLM -> TTS latency tracking
"""

from __future__ import annotations

import logging
import os
import sys
import time


# --- Color codes ---
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[1;31m",  # Bold red
    "RESET": "\033[0m",
    "DIM": "\033[2m",
    "BOLD": "\033[1m",
}

# Pipeline stage colors (for the label, not the whole line)
STAGE_COLORS = {
    "STT": "\033[36m",  # Cyan
    "Memory": "\033[35m",  # Magenta
    "Skill": "\033[35m",  # Magenta
    "LLM": "\033[34m",  # Blue
    "TTS": "\033[33m",  # Yellow
    "Tool": "\033[32m",  # Green
}


class ColorFormatter(logging.Formatter):
    """Colorized log formatter for terminal output."""

    def __init__(self, use_color: bool = True):
        super().__init__(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)

        # Colorize level name
        level_color = COLORS.get(record.levelname, "")
        reset = COLORS["RESET"]
        dim = COLORS["DIM"]

        # Save originals
        orig_levelname = record.levelname
        orig_name = record.name
        orig_asctime = self.formatTime(record, self.datefmt)

        # Build colored output
        record.levelname = f"{level_color}{record.levelname}{reset}"
        record.name = f"{dim}{record.name}{reset}"

        result = super().format(record)

        # Restore
        record.levelname = orig_levelname
        record.name = orig_name

        return result


class PipelineTimer:
    """Tracks timing across pipeline stages for a single request.

    Usage:
        timer = PipelineTimer()
        timer.mark("stt")
        # ... do STT ...
        timer.mark("memory")
        # ... do memory ...
        timer.mark("llm")
        # ... do LLM ...
        timer.mark("tts_done")
        timer.summary()  # -> "STT: 0.1s | Memory: 0.3s | LLM: 2.1s | TTS: 8.2s | Total: 10.7s"
    """

    def __init__(self):
        self._marks: list[tuple[str, float]] = []
        self._start = time.monotonic()

    def mark(self, stage: str) -> None:
        """Record a timestamp for a pipeline stage completion."""
        self._marks.append((stage, time.monotonic()))

    def elapsed(self, stage: str) -> float | None:
        """Get elapsed time for a specific stage (time between previous mark and this one)."""
        for i, (name, ts) in enumerate(self._marks):
            if name == stage:
                prev_ts = self._marks[i - 1][1] if i > 0 else self._start
                return ts - prev_ts
        return None

    def total(self) -> float:
        """Total elapsed time from start."""
        return time.monotonic() - self._start

    def summary(self) -> str:
        """Human-readable timing summary."""
        parts = []
        for i, (name, ts) in enumerate(self._marks):
            prev_ts = self._marks[i - 1][1] if i > 0 else self._start
            elapsed = ts - prev_ts
            parts.append(f"{name}: {elapsed:.1f}s")
        parts.append(f"Total: {self.total():.1f}s")
        return " | ".join(parts)


def _should_use_color() -> bool:
    """Auto-detect color support."""
    env_val = os.getenv("AETHER_LOG_COLOR", "auto").lower()
    if env_val == "true":
        return True
    if env_val == "false":
        return False
    # Auto: use color if stdout is a TTY
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def setup_logging() -> None:
    """Configure logging for the entire application.

    Call this once at startup, before any other imports that use logging.
    Reads AETHER_LOG_LEVEL (default: INFO) and AETHER_LOG_COLOR (default: auto).
    """
    level_name = os.getenv("AETHER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    use_color = _should_use_color()

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers (avoid duplicate output)
    root.handlers.clear()

    # Console handler with our formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(ColorFormatter(use_color=use_color))
    root.addHandler(handler)

    # --- Suppress noisy third-party loggers ---
    # These flood the console with HTTP request/response details
    for noisy_logger in [
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "openai",
        "openai._base_client",
        "deepgram",
        "websockets",
        "uvicorn.access",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Uvicorn error logger stays at configured level (startup messages are useful)
    logging.getLogger("uvicorn.error").setLevel(level)

    logger = logging.getLogger("aether")
    logger.debug("Logging configured (level=%s, color=%s)", level_name, use_color)
