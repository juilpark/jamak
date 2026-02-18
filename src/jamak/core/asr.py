from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from jamak.infra.config import AppConfig
from jamak.schemas.segment import SpeechSegment
from jamak.vendor.qwen3_asr.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

QWEN_ASR_SAMPLE_RATE = 16_000
MIN_SEGMENT_SECONDS = 0.30
ASR_TEXT_TAG = "<asr_text>"
LANGUAGE_PREFIX = "language "

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


@dataclass(frozen=True)
class ASRSegmentResult:
    language: str
    text: str
    raw: str


@dataclass(frozen=True)
class _ASRRuntime:
    model: Qwen3ASRForConditionalGeneration
    processor: Qwen3ASRProcessor


def _normalize_language_name(language: str) -> str:
    s = language.strip()
    return s[:1].upper() + s[1:].lower() if s else s


def _parse_asr_output(raw: str, forced_language: str | None) -> ASRSegmentResult:
    content = (raw or "").strip()
    if not content:
        return ASRSegmentResult(language="", text="", raw=raw)
    if forced_language:
        return ASRSegmentResult(language=forced_language, text=content, raw=raw)
    if ASR_TEXT_TAG not in content:
        return ASRSegmentResult(language="", text=content, raw=raw)

    meta, text = content.split(ASR_TEXT_TAG, 1)
    language = ""
    for line in meta.splitlines():
        candidate = line.strip()
        if not candidate.lower().startswith(LANGUAGE_PREFIX):
            continue
        value = candidate[len(LANGUAGE_PREFIX) :].strip()
        if value and value.lower() != "none":
            language = _normalize_language_name(value)
        break
    return ASRSegmentResult(language=language, text=text.strip(), raw=raw)


def _build_prompt(processor: Qwen3ASRProcessor, forced_language: str | None) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if forced_language:
        return prompt + f"language {forced_language}<asr_text>"
    return prompt


def _runtime_device(config: AppConfig) -> tuple[str, torch.dtype]:
    if config.device == "cuda" and torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


@lru_cache(maxsize=2)
def _load_runtime(
    model_id: str,
    cache_dir: str,
    device: str,
    dtype_name: str,
) -> _ASRRuntime:
    dtype = getattr(torch, dtype_name)
    model = AutoModel.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=dtype,
    )
    model.eval()
    if device != "cpu":
        model.to(device)
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        fix_mistral_regex=True,
    )
    # Avoid repetitive generation warnings by setting a deterministic pad token.
    pad_id = getattr(model.generation_config, "pad_token_id", None)
    if pad_id is None:
        tokenizer = getattr(processor, "tokenizer", None)
        tokenizer_pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None
        eos_id = getattr(model.generation_config, "eos_token_id", None)
        model.generation_config.pad_token_id = (
            tokenizer_pad_id if tokenizer_pad_id is not None else eos_id
        )
    return _ASRRuntime(model=model, processor=processor)


def _read_audio(audio_path: Path) -> np.ndarray:
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    elif not isinstance(audio, np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)
    if sr != QWEN_ASR_SAMPLE_RATE:
        raise RuntimeError(
            f"Expected {QWEN_ASR_SAMPLE_RATE}Hz audio for ASR, got {sr}Hz at {audio_path}."
        )
    return audio.astype(np.float32, copy=False)


def _slice_segment(audio: np.ndarray, segment: SpeechSegment) -> np.ndarray:
    start = max(0, int(segment.start * QWEN_ASR_SAMPLE_RATE))
    end = min(audio.shape[0], int(segment.end * QWEN_ASR_SAMPLE_RATE))
    if end <= start:
        end = min(audio.shape[0], start + 1)
    clip = audio[start:end]
    min_samples = int(MIN_SEGMENT_SECONDS * QWEN_ASR_SAMPLE_RATE)
    if clip.shape[0] < min_samples:
        clip = np.pad(clip, (0, min_samples - clip.shape[0]), mode="constant")
    return clip.astype(np.float32, copy=False)


def transcribe_segments(
    audio_path: Path,
    *,
    segments: list[SpeechSegment],
    config: AppConfig,
    language: str | None,
) -> list[ASRSegmentResult]:
    if not segments:
        return []

    forced_language = _normalize_language_name(language) if language else None
    device, dtype = _runtime_device(config)
    runtime = _load_runtime(
        model_id=config.asr_model_id,
        cache_dir=str(config.hf_cache),
        device=device,
        dtype_name=dtype.__str__().replace("torch.", ""),
    )

    audio = _read_audio(audio_path)
    clips = [_slice_segment(audio, segment) for segment in segments]
    prompts = [_build_prompt(runtime.processor, forced_language) for _ in clips]
    tokenizer = getattr(runtime.processor, "tokenizer", None)
    pad_token_id = (
        getattr(tokenizer, "pad_token_id", None)
        if tokenizer is not None
        else None
    )
    if pad_token_id is None:
        pad_token_id = getattr(runtime.model.generation_config, "eos_token_id", None)

    outputs: list[ASRSegmentResult] = []
    batch_size = config.asr_batch_size
    for idx in range(0, len(clips), batch_size):
        sub_clips = clips[idx : idx + batch_size]
        sub_prompts = prompts[idx : idx + batch_size]
        inputs = runtime.processor(
            text=sub_prompts,
            audio=sub_clips,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(runtime.model.device).to(runtime.model.dtype)
        with torch.no_grad():
            generated = runtime.model.generate(
                **inputs,
                max_new_tokens=config.asr_max_new_tokens,
                pad_token_id=pad_token_id,
            )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        decoded = runtime.processor.batch_decode(
            sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        outputs.extend(_parse_asr_output(raw, forced_language) for raw in decoded)

    return outputs
