from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import unicodedata

import numpy as np
import soundfile as sf
import torch

from jamak.infra.config import AppConfig
from jamak.schemas.segment import SpeechSegment
from jamak.schemas.subtitle import SubtitleCue
from jamak.vendor.qwen3_asr.transformers_backend import (
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

QWEN_ALIGN_SAMPLE_RATE = 16_000
MIN_SEGMENT_SECONDS = 0.20
TIMESTAMP_TAG_PAIR = "<timestamp><timestamp>"
AUDIO_INPUT_MARKER = "<|audio_start|><|audio_pad|><|audio_end|>"


@dataclass(frozen=True)
class AlignedWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class AlignedSegment:
    start: float
    end: float
    text: str
    language: str
    aligned: bool
    words: tuple[AlignedWord, ...]


@dataclass(frozen=True)
class AlignmentRunResult:
    backend: str
    message: str
    segments: list[AlignedSegment]

    @property
    def total_text_segments(self) -> int:
        return sum(1 for item in self.segments if item.text.strip())

    @property
    def aligned_text_segments(self) -> int:
        return sum(1 for item in self.segments if item.text.strip() and item.aligned)


@dataclass(frozen=True)
class _AlignRuntime:
    model: Qwen3ASRForConditionalGeneration
    processor: Qwen3ASRProcessor
    timestamp_token_id: int
    timestamp_segment_time: float


def _normalize_language(language: str | None) -> str:
    if not language:
        return ""
    value = language.strip()
    if not value:
        return ""
    return value[:1].upper() + value[1:].lower()


def _runtime_device(config: AppConfig) -> tuple[str, torch.dtype]:
    if config.device in {"auto", "cuda"} and torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if config.device == "mps" and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


@lru_cache(maxsize=2)
def _load_runtime(
    model_id: str,
    cache_dir: str,
    device: str,
    dtype_name: str,
) -> _AlignRuntime:
    dtype = getattr(torch, dtype_name)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=dtype,
    )
    model.eval()
    if device != "cpu":
        model.to(device)
    processor = Qwen3ASRProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        fix_mistral_regex=True,
    )
    token_id = getattr(model.config, "timestamp_token_id", None)
    segment_time = getattr(model.config, "timestamp_segment_time", None)
    if token_id is None or segment_time is None:
        raise RuntimeError(
            "Forced aligner config is missing timestamp_token_id/timestamp_segment_time."
        )
    return _AlignRuntime(
        model=model,
        processor=processor,
        timestamp_token_id=int(token_id),
        timestamp_segment_time=float(segment_time),
    )


def _read_audio(audio_path: Path) -> np.ndarray:
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    elif not isinstance(audio, np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)
    if sr != QWEN_ALIGN_SAMPLE_RATE:
        raise RuntimeError(
            f"Expected {QWEN_ALIGN_SAMPLE_RATE}Hz audio for aligner, got {sr}Hz at {audio_path}."
        )
    return audio.astype(np.float32, copy=False)


def _slice_segment(audio: np.ndarray, segment: SpeechSegment) -> np.ndarray:
    start = max(0, int(segment.start * QWEN_ALIGN_SAMPLE_RATE))
    end = min(audio.shape[0], int(segment.end * QWEN_ALIGN_SAMPLE_RATE))
    if end <= start:
        end = min(audio.shape[0], start + 1)
    clip = audio[start:end]
    min_samples = int(MIN_SEGMENT_SECONDS * QWEN_ALIGN_SAMPLE_RATE)
    if clip.shape[0] < min_samples:
        clip = np.pad(clip, (0, min_samples - clip.shape[0]), mode="constant")
    return clip.astype(np.float32, copy=False)


def _is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def _is_cjk_or_japanese_or_korean(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x9FFF
        or 0x3040 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0xAC00 <= code <= 0xD7AF
    )


def _clean_token(token: str) -> str:
    return "".join(ch for ch in token if _is_kept_char(ch))


def _tokenize_for_alignment(text: str) -> list[str]:
    tokens: list[str] = []
    latin_buffer: list[str] = []

    def flush() -> None:
        nonlocal latin_buffer
        if not latin_buffer:
            return
        cleaned = _clean_token("".join(latin_buffer))
        if cleaned:
            tokens.append(cleaned)
        latin_buffer = []

    for ch in text:
        if _is_cjk_or_japanese_or_korean(ch):
            flush()
            if _is_kept_char(ch):
                tokens.append(ch)
            continue
        if _is_kept_char(ch):
            latin_buffer.append(ch)
        else:
            flush()

    flush()
    return tokens


def _build_timestamp_prompt(words: list[str]) -> str:
    if not words:
        return AUDIO_INPUT_MARKER
    return AUDIO_INPUT_MARKER + TIMESTAMP_TAG_PAIR.join(words) + TIMESTAMP_TAG_PAIR


def _fix_monotonic(values: list[float]) -> list[float]:
    if not values:
        return values
    fixed = [values[0]]
    for value in values[1:]:
        fixed.append(max(fixed[-1], value))
    return fixed


def _build_word_timestamps(
    *,
    words: list[str],
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    timestamp_token_id: int,
    timestamp_segment_time: float,
) -> list[tuple[str, float, float]]:
    if not words:
        return []
    masked = output_ids[input_ids == timestamp_token_id]
    timestamps_ms = (masked.to(torch.float32) * timestamp_segment_time).cpu().tolist()
    required = len(words) * 2
    if len(timestamps_ms) < required:
        pad_value = timestamps_ms[-1] if timestamps_ms else 0.0
        timestamps_ms.extend([pad_value] * (required - len(timestamps_ms)))
    elif len(timestamps_ms) > required:
        timestamps_ms = timestamps_ms[:required]
    timestamps_ms = _fix_monotonic(timestamps_ms)

    pairs: list[tuple[str, float, float]] = []
    for index, word in enumerate(words):
        start_ms = float(timestamps_ms[index * 2])
        end_ms = float(timestamps_ms[index * 2 + 1])
        if end_ms < start_ms:
            end_ms = start_ms
        pairs.append((word, start_ms / 1000.0, end_ms / 1000.0))
    return pairs


def _fallback_segments(
    segments: list[SpeechSegment],
    texts: list[str],
    languages: list[str],
) -> list[AlignedSegment]:
    items: list[AlignedSegment] = []
    for index, segment in enumerate(segments):
        text = texts[index].strip() if index < len(texts) else ""
        language = languages[index] if index < len(languages) else ""
        start = max(0.0, float(segment.start))
        end = max(start + 0.01, float(segment.end))
        items.append(
            AlignedSegment(
                start=start,
                end=end,
                text=text,
                language=language,
                aligned=False,
                words=(),
            )
        )
    return items


def _move_inputs(
    batch_inputs: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    for key, value in list(batch_inputs.items()):
        if not isinstance(value, torch.Tensor):
            continue
        moved = value.to(device)
        if torch.is_floating_point(moved):
            moved = moved.to(dtype=dtype)
        batch_inputs[key] = moved
    return batch_inputs


def align_segments(
    audio_path: Path,
    *,
    segments: list[SpeechSegment],
    texts: list[str],
    languages: list[str],
    config: AppConfig,
) -> AlignmentRunResult:
    normalized_languages = [_normalize_language(language) for language in languages]
    aligned_segments = _fallback_segments(segments, texts, normalized_languages)
    if not segments:
        return AlignmentRunResult(
            backend="fallback",
            message="No segments to align.",
            segments=aligned_segments,
        )
    if not any(item.text.strip() for item in aligned_segments):
        return AlignmentRunResult(
            backend="fallback",
            message="No ASR text to align. Used VAD segment timestamps.",
            segments=aligned_segments,
        )

    device_name, dtype = _runtime_device(config)
    try:
        runtime = _load_runtime(
            model_id=config.align_model_id,
            cache_dir=str(config.hf_cache),
            device=device_name,
            dtype_name=str(dtype).replace("torch.", ""),
        )
        audio = _read_audio(audio_path)
    except Exception as exc:
        return AlignmentRunResult(
            backend="fallback",
            message=f"Forced aligner unavailable ({exc}). Used VAD segment timestamps.",
            segments=aligned_segments,
        )

    targets: list[tuple[int, np.ndarray, list[str]]] = []
    for index, segment in enumerate(segments):
        if index >= len(aligned_segments):
            break
        text = aligned_segments[index].text.strip()
        if not text:
            continue
        words = _tokenize_for_alignment(text)
        if not words:
            continue
        targets.append((index, _slice_segment(audio, segment), words))

    if not targets:
        return AlignmentRunResult(
            backend="fallback",
            message="Forced aligner skipped: tokenization produced no alignable words.",
            segments=aligned_segments,
        )

    batch_size = max(1, config.align_batch_size)
    success_count = 0
    for offset in range(0, len(targets), batch_size):
        window = targets[offset : offset + batch_size]
        indexes = [item[0] for item in window]
        clips = [item[1] for item in window]
        words_list = [item[2] for item in window]
        prompts = [_build_timestamp_prompt(words) for words in words_list]
        batch_inputs = runtime.processor(
            text=prompts,
            audio=clips,
            return_tensors="pt",
            padding=True,
        )
        moved_inputs = _move_inputs(
            dict(batch_inputs),
            device=runtime.model.device,
            dtype=runtime.model.dtype,
        )
        with torch.inference_mode():
            logits = runtime.model.thinker(**moved_inputs).logits
        predictions = logits.argmax(dim=-1)

        for row, seg_index in enumerate(indexes):
            base_segment = segments[seg_index]
            word_pairs = _build_word_timestamps(
                words=words_list[row],
                input_ids=moved_inputs["input_ids"][row],
                output_ids=predictions[row],
                timestamp_token_id=runtime.timestamp_token_id,
                timestamp_segment_time=runtime.timestamp_segment_time,
            )
            if not word_pairs:
                continue
            absolute_words: list[AlignedWord] = []
            base_start = float(base_segment.start)
            for word, rel_start, rel_end in word_pairs:
                start = round(max(base_start, base_start + rel_start), 3)
                end = round(max(start + 0.001, base_start + rel_end), 3)
                absolute_words.append(AlignedWord(text=word, start=start, end=end))
            if not absolute_words:
                continue
            current = aligned_segments[seg_index]
            aligned_segments[seg_index] = AlignedSegment(
                start=absolute_words[0].start,
                end=max(absolute_words[-1].end, absolute_words[0].start + 0.01),
                text=current.text,
                language=current.language,
                aligned=True,
                words=tuple(absolute_words),
            )
            success_count += 1

    if success_count == 0:
        return AlignmentRunResult(
            backend="fallback",
            message="Forced aligner returned empty alignment. Used VAD segment timestamps.",
            segments=aligned_segments,
        )

    total_text_segments = sum(1 for item in aligned_segments if item.text.strip())
    fallback_count = total_text_segments - success_count
    if fallback_count == 0:
        message = f"Qwen forced aligner aligned {success_count}/{total_text_segments} text segments."
    else:
        message = (
            f"Qwen forced aligner aligned {success_count}/{total_text_segments} text segments. "
            f"{fallback_count} segments used VAD timestamp fallback."
        )
    return AlignmentRunResult(
        backend="qwen-forced-aligner",
        message=message,
        segments=aligned_segments,
    )


def cues_from_aligned_segments(segments: list[AlignedSegment]) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        start = max(0.0, float(segment.start))
        end = max(start + 0.01, float(segment.end))
        cues.append(SubtitleCue(start=start, end=end, text=text))
    return cues
