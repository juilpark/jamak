from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jamak.infra.config import AppConfig
from jamak.infra.storage import ensure_directory


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
    planned_outputs: int
    status: str
    message: str


def transcribe_file(request: TranscribeRequest) -> TranscribeResult:
    """Phase 0 skeleton for a single-file transcription run."""
    ensure_directory(request.output_dir)
    output_path = request.output_dir / f"{request.input_path.stem}.srt"
    return TranscribeResult(
        input_path=request.input_path,
        output_path=output_path,
        status="planned",
        message="Pipeline skeleton ready. Implement VAD/ASR/Align steps in Phase 1.",
    )


def batch_transcribe(request: BatchRequest) -> BatchResult:
    """Phase 0 skeleton for batch runs."""
    ensure_directory(request.output_dir)
    files = [
        path
        for path in sorted(request.input_dir.glob(request.glob_pattern))
        if path.is_file()
    ]
    return BatchResult(
        total=len(files),
        planned_outputs=len(files),
        status="planned",
        message="Batch skeleton ready. Implement VAD/ASR/Align steps in Phase 1.",
    )

