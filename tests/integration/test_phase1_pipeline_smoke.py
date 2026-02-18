from __future__ import annotations

from pathlib import Path

import pytest

from jamak.core.align import AlignedSegment, AlignmentRunResult
from jamak.core.asr import ASRSegmentResult
from jamak.core.pipeline import TranscribeRequest, transcribe_file
from jamak.core.vad import VadRunResult
from jamak.infra.config import build_app_config
from jamak.infra.ffmpeg import get_ffmpeg_path
from jamak.schemas.segment import SpeechSegment


def test_phase1_pipeline_smoke_with_real_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    if get_ffmpeg_path() is None:
        pytest.skip("ffmpeg is not available")

    sample_audio = Path("test_audio.mp3")
    if not sample_audio.exists():
        pytest.skip("test_audio.mp3 is not available")

    monkeypatch.setattr(
        "jamak.core.pipeline.detect_speech_segments",
        lambda **_: VadRunResult(
            backend="fallback",
            segments=[SpeechSegment(start=0.0, end=2.0, confidence=None)],
            message="Fallback VAD selected by test.",
        ),
    )
    monkeypatch.setattr(
        "jamak.core.pipeline.transcribe_segments",
        lambda **_: [
            ASRSegmentResult(
                language="Japanese",
                text="これは統合テストです。",
                raw="language Japanese<asr_text>これは統合テストです。",
            )
        ],
    )
    monkeypatch.setattr(
        "jamak.core.pipeline.align_segments",
        lambda **_: AlignmentRunResult(
            backend="qwen-forced-aligner",
            message="Qwen forced aligner aligned 1/1 text segments.",
            segments=[
                AlignedSegment(
                    start=0.1,
                    end=1.9,
                    text="これは統合テストです。",
                    language="Japanese",
                    aligned=True,
                    words=(),
                )
            ],
        ),
    )

    output_dir = tmp_path / "outputs"
    result = transcribe_file(
        TranscribeRequest(
            input_path=sample_audio,
            output_dir=output_dir,
            language=None,
            config=build_app_config(vad_backend="fallback"),
        )
    )

    assert result.status == "done"
    assert result.output_path.exists()
    assert result.audio_path.exists()
    assert result.run_path.exists()
    assert result.segments_path.exists()
    assert result.log_path.exists()
    assert "これは統合テストです。" in result.output_path.read_text(encoding="utf-8")
