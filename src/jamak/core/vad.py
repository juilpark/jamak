from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jamak.infra.config import AppConfig
from jamak.schemas.segment import SpeechSegment


@dataclass(frozen=True)
class VadRunResult:
    backend: str
    segments: list[SpeechSegment]
    message: str


def _full_range_segment(duration_seconds: float | None) -> SpeechSegment:
    end = duration_seconds if duration_seconds and duration_seconds > 0 else 1.0
    return SpeechSegment(start=0.0, end=float(end), confidence=None)


def _parse_timestamps(
    timestamps: list[Any], probs: list[Any] | None = None
) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    for idx, item in enumerate(timestamps):
        start: float | None = None
        end: float | None = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start = float(item[0])
            end = float(item[1])
        elif isinstance(item, dict):
            if "start" in item and "end" in item:
                start = float(item["start"])
                end = float(item["end"])
        if start is None or end is None or end <= start:
            continue
        confidence: float | None = None
        if probs and idx < len(probs):
            try:
                confidence = float(probs[idx])
            except (TypeError, ValueError):
                confidence = None
        segments.append(SpeechSegment(start=start, end=end, confidence=confidence))
    return segments


@lru_cache(maxsize=2)
def _load_firered_model(model_dir: str, use_gpu: bool) -> Any:
    try:
        from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
    except ImportError as exc:
        raise RuntimeError(
            "FireRedVAD backend is not installed. Install FireRedASR2S first."
        ) from exc
    config = FireRedVadConfig(use_gpu=use_gpu)
    return FireRedVad.from_pretrained(model_dir, config)


def _detect_with_firered(
    audio_path: Path, model_dir: Path, use_gpu: bool
) -> list[SpeechSegment]:
    model = _load_firered_model(str(model_dir), use_gpu)
    result, probs = model.detect(str(audio_path))
    timestamps = result.get("timestamps", []) if isinstance(result, dict) else []
    if not isinstance(timestamps, list):
        timestamps = []
    prob_list = probs if isinstance(probs, list) else None
    return _parse_timestamps(timestamps, prob_list)


def detect_speech_segments(
    audio_path: Path,
    *,
    config: AppConfig,
    duration_seconds: float | None,
) -> VadRunResult:
    if config.vad_backend in {"auto", "firered"}:
        try:
            segments = _detect_with_firered(
                audio_path=audio_path,
                model_dir=config.vad_model_dir,
                use_gpu=config.device == "cuda",
            )
            if segments:
                return VadRunResult(
                    backend="firered",
                    segments=segments,
                    message=f"FireRedVAD detected {len(segments)} segments.",
                )
            fallback = [_full_range_segment(duration_seconds)]
            return VadRunResult(
                backend="fallback",
                segments=fallback,
                message="FireRedVAD returned no segments. Used full-range fallback.",
            )
        except Exception as exc:
            fallback = [_full_range_segment(duration_seconds)]
            return VadRunResult(
                backend="fallback",
                segments=fallback,
                message=f"FireRedVAD unavailable ({exc}). Used full-range fallback.",
            )

    fallback = [_full_range_segment(duration_seconds)]
    return VadRunResult(
        backend="fallback",
        segments=fallback,
        message="Fallback VAD selected by configuration.",
    )

