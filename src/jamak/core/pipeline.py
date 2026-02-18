from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jamak.core.asr import transcribe_segments
from jamak.core.subtitle import write_srt
from jamak.core.vad import detect_speech_segments
from jamak.infra.config import AppConfig
from jamak.infra.ffmpeg import extract_audio, probe_duration_seconds
from jamak.infra.storage import ensure_directory, write_json
from jamak.schemas.segment import SpeechSegment
from jamak.schemas.subtitle import SubtitleCue


@dataclass(frozen=True)
class TranscribeRequest:
    input_path: Path
    output_dir: Path
    language: str | None
    config: AppConfig


@dataclass(frozen=True)
class TranscribeResult:
    input_path: Path
    output_path: Path
    audio_path: Path
    run_path: Path
    segments_path: Path
    status: str
    message: str


@dataclass(frozen=True)
class BatchRequest:
    input_dir: Path
    output_dir: Path
    glob_pattern: str
    language: str | None
    config: AppConfig


@dataclass(frozen=True)
class BatchResult:
    total: int
    succeeded: int
    partial: int
    failed: int
    status: str
    message: str


def _build_artifact_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path, Path]:
    stem = input_path.stem
    output_path = output_dir / f"{stem}.srt"
    meta_dir = output_dir / ".jamak" / stem
    audio_path = meta_dir / "audio.wav"
    run_path = meta_dir / "run.json"
    segments_path = meta_dir / "segments.json"
    return output_path, audio_path, run_path, segments_path


def _build_subtitle_cues(
    segments: list[SpeechSegment],
    texts: list[str],
    duration_seconds: float | None,
) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for index, segment in enumerate(segments):
        if index >= len(texts):
            break
        text = texts[index].strip()
        if not text:
            continue
        start = max(0.0, float(segment.start))
        end = max(start + 0.01, float(segment.end))
        cues.append(SubtitleCue(start=start, end=end, text=text))
    if cues:
        return cues
    if segments:
        first = segments[0]
        return [
            SubtitleCue(
                start=float(first.start),
                end=max(float(first.end), float(first.start) + 0.01),
                text="[Phase 2] ASR produced empty text. Alignment pending.",
            )
        ]
    fallback_end = 3.0 if duration_seconds is None else max(1.0, min(duration_seconds, 10.0))
    return [
        SubtitleCue(
            start=0.0,
            end=fallback_end,
            text="[Phase 2] No speech segments found. Alignment pending.",
        )
    ]


def transcribe_file(request: TranscribeRequest) -> TranscribeResult:
    """Phase 1 step: extract audio and emit baseline artifacts."""
    ensure_directory(request.output_dir)
    output_path, audio_path, run_path, segments_path = _build_artifact_paths(
        request.input_path, request.output_dir
    )
    ensure_directory(audio_path.parent)
    try:
        extract_audio(request.input_path, audio_path)
        duration_seconds = probe_duration_seconds(audio_path)
        vad_result = detect_speech_segments(
            audio_path=audio_path,
            config=request.config,
            duration_seconds=duration_seconds,
        )
        asr_results = transcribe_segments(
            audio_path=audio_path,
            segments=vad_result.segments,
            config=request.config,
            language=request.language,
        )
        write_srt(
            _build_subtitle_cues(
                vad_result.segments,
                [result.text for result in asr_results],
                duration_seconds,
            ),
            output_path,
        )
        write_json(
            segments_path,
            {
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "confidence": segment.confidence,
                        "text": (
                            asr_results[index].text if index < len(asr_results) else ""
                        ),
                        "language": (
                            asr_results[index].language if index < len(asr_results) else ""
                        ),
                    }
                    for index, segment in enumerate(vad_result.segments)
                ],
                "duration_seconds": duration_seconds,
                "vad_backend": vad_result.backend,
                "vad_message": vad_result.message,
                "asr_model_id": request.config.asr_model_id,
                "note": "ASR done. Forced alignment pending.",
            },
        )
        write_json(
            run_path,
            {
                "status": "partial",
                "phase": "phase1-audio-extract",
                "input_path": str(request.input_path),
                "audio_path": str(audio_path),
                "output_srt_path": str(output_path),
                "segments_path": str(segments_path),
                "language": request.language,
                "device": request.config.device,
                "duration_seconds": duration_seconds,
                "vad_backend": vad_result.backend,
                "vad_segments": len(vad_result.segments),
                "vad_message": vad_result.message,
                "asr_model_id": request.config.asr_model_id,
                "asr_results": len(asr_results),
                "asr_max_new_tokens": request.config.asr_max_new_tokens,
                "asr_batch_size": request.config.asr_batch_size,
            },
        )
        return TranscribeResult(
            input_path=request.input_path,
            output_path=output_path,
            audio_path=audio_path,
            run_path=run_path,
            segments_path=segments_path,
            status="partial",
            message=(
                "Audio extraction and ASR complete. "
                f"{vad_result.message} Forced alignment integration pending."
            ),
        )
    except Exception as exc:
        write_json(
            run_path,
            {
                "status": "failed",
                "phase": "phase1-audio-extract",
                "input_path": str(request.input_path),
                "audio_path": str(audio_path),
                "output_srt_path": str(output_path),
                "segments_path": str(segments_path),
                "language": request.language,
                "device": request.config.device,
                "error": str(exc),
            },
        )
        return TranscribeResult(
            input_path=request.input_path,
            output_path=output_path,
            audio_path=audio_path,
            run_path=run_path,
            segments_path=segments_path,
            status="failed",
            message=str(exc),
        )


def batch_transcribe(request: BatchRequest) -> BatchResult:
    """Phase 1 step: batch audio extraction and artifact generation."""
    ensure_directory(request.output_dir)
    files = [
        path
        for path in sorted(request.input_dir.glob(request.glob_pattern))
        if path.is_file()
    ]
    if not files:
        return BatchResult(
            total=0,
            succeeded=0,
            partial=0,
            failed=0,
            status="failed",
            message="No input files matched the glob pattern.",
        )
    succeeded = 0
    partial = 0
    failed = 0
    for input_path in files:
        result = transcribe_file(
            TranscribeRequest(
                input_path=input_path,
                output_dir=request.output_dir,
                language=request.language,
                config=request.config,
            )
        )
        if result.status == "done":
            succeeded += 1
        elif result.status == "partial":
            partial += 1
        else:
            failed += 1

    if failed and not (succeeded or partial):
        status = "failed"
    elif failed:
        status = "partial"
    elif partial:
        status = "partial"
    else:
        status = "done"
    message = (
        f"Batch complete: total={len(files)} succeeded={succeeded} "
        f"partial={partial} failed={failed}."
    )
    return BatchResult(
        total=len(files),
        succeeded=succeeded,
        partial=partial,
        failed=failed,
        status=status,
        message=message,
    )
