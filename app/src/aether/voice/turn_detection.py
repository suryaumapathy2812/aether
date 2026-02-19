from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from aether.core.config import TurnDetectionConfig

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6
MAX_HISTORY_TOKENS = 128

_DETECTOR_CACHE: dict[tuple[str, str, str], "LiveKitTurnDetector"] = {}


@dataclass(frozen=True)
class TurnDecision:
    probability: float
    threshold: float
    likely_end_of_turn: bool
    recommended_delay_s: float


class LiveKitTurnDetector:
    """LiveKit turn detector model adapter (local ONNX inference)."""

    def __init__(self, cfg: TurnDetectionConfig) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._cfg = cfg
        model_dir = _resolve_model_dir(cfg.model_dir)
        model_path = model_dir / "onnx" / cfg.model_filename
        languages_path = model_dir / "languages.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Turn detector model not found: {model_path}")
        if not languages_path.exists():
            raise FileNotFoundError(
                f"Turn detector language config not found: {languages_path}"
            )

        with languages_path.open("r", encoding="utf-8") as f:
            self._languages = json.load(f)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=sess_options,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            truncation_side="left",
            local_files_only=True,
        )

        # Warm up tokenizer + ONNX session once so first live turn
        # does not hit an aggressive inference timeout.
        self._warmup()

    async def predict_delay(
        self,
        history: list[dict[str, str]],
        user_text: str,
        language: str | None,
    ) -> TurnDecision:
        messages = history[-(MAX_HISTORY_TURNS - 1) :] + [
            {"role": "user", "content": user_text}
        ]
        probability = await asyncio.wait_for(
            asyncio.to_thread(self._predict_probability_sync, messages),
            timeout=self._cfg.inference_timeout_seconds,
        )

        threshold = self._resolve_threshold(language)
        likely_end = probability >= threshold
        delay = (
            self._cfg.min_endpointing_delay
            if likely_end
            else self._cfg.max_endpointing_delay
        )
        return TurnDecision(
            probability=probability,
            threshold=threshold,
            likely_end_of_turn=likely_end,
            recommended_delay_s=delay,
        )

    def _predict_probability_sync(self, messages: list[dict[str, str]]) -> float:
        import numpy as np

        text = self._format_chat_context(messages)
        inputs = self._tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="np",
            max_length=MAX_HISTORY_TOKENS,
            truncation=True,
        )
        outputs = self._session.run(
            None, {"input_ids": inputs["input_ids"].astype("int64")}
        )
        return float(outputs[0].flatten()[-1])

    def _warmup(self) -> None:
        try:
            self._predict_probability_sync(
                [
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "hello"},
                ]
            )
        except Exception as e:
            logger.debug("Turn detector warmup failed: %s", e)

    def _resolve_threshold(self, language: str | None) -> float:
        if not language:
            return 0.5
        lang = language.lower()
        data = self._languages.get(lang)
        if data is None and "-" in lang:
            data = self._languages.get(lang.split("-")[0])
        if isinstance(data, dict) and "threshold" in data:
            return float(data["threshold"])
        return 0.5

    def _normalize_text(self, text: str) -> str:
        # Normalize unicode and collapse whitespace only.
        # Do NOT strip punctuation — the model was trained on punctuated text
        # and relies on sentence-ending signals (. ! ?) to score end-of-turn.
        text = unicodedata.normalize("NFKC", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    def _format_chat_context(self, messages: list[dict[str, str]]) -> str:
        compact: list[dict[str, str]] = []
        for msg in messages:
            content = self._normalize_text(msg.get("content", ""))
            if not content:
                continue
            role = msg.get("role", "user")
            if compact and compact[-1]["role"] == role:
                compact[-1]["content"] += f" {content}"
            else:
                compact.append({"role": role, "content": content})

        convo = self._tokenizer.apply_chat_template(
            compact,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )
        ix = convo.rfind("<|im_end|>")
        return convo if ix == -1 else convo[:ix]


def build_turn_detector(cfg: TurnDetectionConfig) -> LiveKitTurnDetector | None:
    if cfg.mode == "off":
        return None

    cache_key = (cfg.model_dir, cfg.model_repo, cfg.model_revision)
    cached = _DETECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        detector = LiveKitTurnDetector(cfg)
        _DETECTOR_CACHE[cache_key] = detector
        logger.info(
            "Turn detector initialized (mode=%s, model=%s@%s)",
            cfg.mode,
            cfg.model_repo,
            cfg.model_revision,
        )
        return detector
    except Exception as e:
        logger.warning("Failed to initialize turn detector, falling back to off: %s", e)
        return None


def _resolve_model_dir(raw_path: str) -> Path:
    path = raw_path.strip()
    if not path:
        raise ValueError("Turn model dir is empty")
    p = Path(path)
    return p
