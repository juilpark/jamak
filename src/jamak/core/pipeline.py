from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jamak.core.align import align_segments, cues_from_aligned_segments
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
    log_path: Path
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


def _build_artifact_paths(
    input_path: Path, output_dir: Path
) -> tuple[Path, Path, Path, Path, Path]:
    stem = input_path.stem
    output_path = output_dir / f"{stem}.srt"
    meta_dir = output_dir / ".jamak" / stem
    audio_path = meta_dir / "audio.wav"
    run_path = meta_dir / "run.json"
    segments_path = meta_dir / "segments.json"
    log_path = meta_dir / "run.log"
    return output_path, audio_path, run_path, segments_path, log_path


def _write_run_log(log_path: Path, lines: list[str]) -> None:
    log_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _build_subtitle_cues(
    segments: list[SpeechSegment],
    duration_seconds: float | None,
) -> list[SubtitleCue]:
    if segments:
        first = segments[0]
        return [
            SubtitleCue(
                start=float(first.start),
                end=max(float(first.end), float(first.start) + 0.01),
                text="[Phase 1] ASR produced empty text.",
            )
        ]
    fallback_end = 3.0 if duration_seconds is None else max(1.0, min(duration_seconds, 10.0))
    return [
        SubtitleCue(
            start=0.0,
            end=fallback_end,
            text="[Phase 1] No speech segments found.",
        )
    ]


def transcribe_file(request: TranscribeRequest) -> TranscribeResult:
    """Phase 1 step: extract audio and produce aligned subtitle artifacts."""
    ensure_directory(request.output_dir)
    output_path, audio_path, run_path, segments_path, log_path = _build_artifact_paths(
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
        asr_texts = [result.text for result in asr_results]
        asr_languages = [
            result.language if result.language else (request.language or "")
            for result in asr_results
        ]
        align_result = align_segments(
            audio_path=audio_path,
            segments=vad_result.segments,
            texts=asr_texts,
            languages=asr_languages,
            config=request.config,
        )
        cues = cues_from_aligned_segments(align_result.segments)
        if not cues:
            cues = _build_subtitle_cues(
                vad_result.segments,
                duration_seconds,
            )
        write_srt(
            cues,
            output_path,
        )
        total_text_segments = align_result.total_text_segments
        aligned_text_segments = align_result.aligned_text_segments
        is_alignment_complete = (
            total_text_segments > 0 and aligned_text_segments == total_text_segments
        )
        run_status = "done" if is_alignment_complete else "partial"
        write_json(
            segments_path,
            {
                "segments": [
                    {
                        "vad_start": segment.start,
                        "vad_end": segment.end,
                        "start": align_result.segments[index].start,
                        "end": align_result.segments[index].end,
                        "confidence": segment.confidence,
                        "text": align_result.segments[index].text,
                        "language": align_result.segments[index].language,
                        "aligned": align_result.segments[index].aligned,
                        "word_count": len(align_result.segments[index].words),
                        "words": [
                            {
                                "text": word.text,
                                "start": word.start,
                                "end": word.end,
                            }
                            for word in align_result.segments[index].words
                        ],
                    }
                    for index, segment in enumerate(vad_result.segments)
                ],
                "duration_seconds": duration_seconds,
                "vad_backend": vad_result.backend,
                "vad_message": vad_result.message,
                "asr_model_id": request.config.asr_model_id,
                "align_backend": align_result.backend,
                "align_model_id": request.config.align_model_id,
                "align_message": align_result.message,
                "align_total_text_segments": total_text_segments,
                "align_aligned_text_segments": aligned_text_segments,
                "note": "Forced alignment complete.",
            },
        )
        write_json(
            run_path,
            {
                "status": run_status,
                "phase": "phase1-local-cli-mvp",
                "input_path": str(request.input_path),
                "audio_path": str(audio_path),
                "output_srt_path": str(output_path),
                "segments_path": str(segments_path),
                "log_path": str(log_path),
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
                "align_backend": align_result.backend,
                "align_model_id": request.config.align_model_id,
                "align_message": align_result.message,
                "align_total_text_segments": total_text_segments,
                "align_aligned_text_segments": aligned_text_segments,
                "align_batch_size": request.config.align_batch_size,
            },
        )
        _write_run_log(
            log_path,
            [
                f"status={run_status}",
                f"input={request.input_path}",
                f"audio={audio_path}",
                f"output_srt={output_path}",
                f"segments={segments_path}",
                f"vad_backend={vad_result.backend}",
                f"vad_segments={len(vad_result.segments)}",
                f"vad_message={vad_result.message}",
                f"asr_model={request.config.asr_model_id}",
                f"asr_results={len(asr_results)}",
                f"align_backend={align_result.backend}",
                f"align_model={request.config.align_model_id}",
                f"align_total_text_segments={total_text_segments}",
                f"align_aligned_text_segments={aligned_text_segments}",
                f"align_message={align_result.message}",
            ],
        )
        return TranscribeResult(
            input_path=request.input_path,
            output_path=output_path,
            audio_path=audio_path,
            run_path=run_path,
            segments_path=segments_path,
            log_path=log_path,
            status=run_status,
            message=(
                f"VAD and ASR complete. {align_result.message} "
                f"(VAD: {vad_result.message})"
            ),
        )
    except Exception as exc:
        write_json(
            run_path,
            {
                "status": "failed",
                "phase": "phase1-local-cli-mvp",
                "input_path": str(request.input_path),
                "audio_path": str(audio_path),
                "output_srt_path": str(output_path),
                "segments_path": str(segments_path),
                "log_path": str(log_path),
                "language": request.language,
                "device": request.config.device,
                "error": str(exc),
            },
        )
        _write_run_log(
            log_path,
            [
                "status=failed",
                f"input={request.input_path}",
                f"audio={audio_path}",
                f"output_srt={output_path}",
                f"segments={segments_path}",
                f"error={exc}",
            ],
        )
        return TranscribeResult(
            input_path=request.input_path,
            output_path=output_path,
            audio_path=audio_path,
            run_path=run_path,
            segments_path=segments_path,
            log_path=log_path,
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
