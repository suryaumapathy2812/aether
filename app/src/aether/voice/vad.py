from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VADSettings:
    mode: str
    model_path: str
    sample_rate: int
    activation_threshold: float
    deactivation_threshold: float
    min_speech_duration: float
    min_silence_duration: float


class SileroVAD:
    """Minimal Silero ONNX VAD processor.

    Consumes 16kHz mono PCM16 bytes and emits speech boundary events.
    """

    def __init__(self, settings: VADSettings) -> None:
        import onnxruntime as ort

        if settings.sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports only 8kHz or 16kHz")

        model_path = Path(settings.model_path)
        if not model_path.exists() or not model_path.is_file():
            raise FileNotFoundError(f"VAD model not found: {model_path}")

        self._settings = settings
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        if settings.sample_rate == 16000:
            self._window_size = 512
            self._context_size = 64
        else:
            self._window_size = 256
            self._context_size = 32

        self._sample_rate_nd = np.array(settings.sample_rate, dtype=np.int64)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self._rnn_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._input_buffer = np.zeros(
            (1, self._context_size + self._window_size), dtype=np.float32
        )

        self._pending = np.empty(0, dtype=np.int16)
        self._speaking = False
        self._speech_acc = 0.0
        self._silence_acc = 0.0
        self._samples_seen = 0

    def process_pcm16(self, pcm_bytes: bytes) -> list[dict[str, Any]]:
        if not pcm_bytes:
            return []

        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if samples.size == 0:
            return []

        self._pending = np.concatenate((self._pending, samples))
        events: list[dict[str, Any]] = []

        while self._pending.size >= self._window_size:
            window_i16 = self._pending[: self._window_size]
            self._pending = self._pending[self._window_size :]

            prob = self._infer_probability(window_i16)
            window_duration = self._window_size / self._settings.sample_rate
            self._samples_seen += self._window_size

            if prob >= self._settings.activation_threshold or (
                self._speaking and prob > self._settings.deactivation_threshold
            ):
                self._speech_acc += window_duration
                self._silence_acc = 0.0
                if (
                    not self._speaking
                    and self._speech_acc >= self._settings.min_speech_duration
                ):
                    self._speaking = True
                    events.append(self._event("speech_started", prob))
            else:
                self._silence_acc += window_duration
                self._speech_acc = 0.0
                if (
                    self._speaking
                    and self._silence_acc >= self._settings.min_silence_duration
                ):
                    self._speaking = False
                    events.append(self._event("speech_ended", prob))

        return events

    def _infer_probability(self, window_i16: np.ndarray) -> float:
        np.divide(
            window_i16,
            np.iinfo(np.int16).max,
            out=self._input_buffer[:, self._context_size :],
            dtype=np.float32,
        )
        self._input_buffer[:, : self._context_size] = self._context

        outputs = self._session.run(
            None,
            {
                "input": self._input_buffer,
                "state": self._rnn_state,
                "sr": self._sample_rate_nd,
            },
        )
        out, self._rnn_state = outputs
        self._context = self._input_buffer[:, -self._context_size :]
        return float(out.item())

    def _event(self, action: str, probability: float) -> dict[str, Any]:
        return {
            "action": action,
            "probability": probability,
            "samples": self._samples_seen,
            "seconds": self._samples_seen / self._settings.sample_rate,
        }


def build_vad(settings: VADSettings) -> SileroVAD | None:
    if settings.mode == "off":
        return None

    if not settings.model_path:
        logger.warning("VAD enabled but AETHER_VAD_MODEL_PATH is empty; disabling VAD")
        return None

    try:
        vad = SileroVAD(settings)
        logger.info(
            "VAD initialized (mode=%s, sample_rate=%d)",
            settings.mode,
            settings.sample_rate,
        )
        return vad
    except Exception as e:
        logger.warning("Failed to initialize VAD, falling back to off: %s", e)
        return None
