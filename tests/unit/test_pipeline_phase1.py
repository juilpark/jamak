from __future__ import annotations

import json
from pathlib import Path

from jamak.core.pipeline import (
    BatchRequest,
    TranscribeRequest,
    batch_transcribe,
    transcribe_file,
)
from jamak.core.asr import ASRSegmentResult
from jamak.core.vad import VadRunResult
from jamak.infra.config import build_app_config
from jamak.schemas.segment import SpeechSegment


def test_transcribe_file_writes_phase1_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "sample.mp4"
    input_path.write_bytes(b"fake-input")
    output_dir = tmp_path / "outputs"

    def fake_extract_audio(_: Path, output_path: Path, **__: object) -> None:
        output_path.write_bytes(b"RIFF0000WAVE")

    monkeypatch.setattr("jamak.core.pipeline.extract_audio", fake_extract_audio)
    monkeypatch.setattr("jamak.core.pipeline.probe_duration_seconds", lambda _: 2.5)
    monkeypatch.setattr(
        "jamak.core.pipeline.detect_speech_segments",
        lambda **_: VadRunResult(
            backend="firered",
            segments=[SpeechSegment(start=0.1, end=1.4, confidence=0.8)],
            message="FireRedVAD detected 1 segments.",
        ),
    )
    monkeypatch.setattr(
        "jamak.core.pipeline.transcribe_segments",
        lambda **_: [
            ASRSegmentResult(
                language="Korean",
                text="테스트 문장입니다.",
                raw="language Korean<asr_text>테스트 문장입니다.",
            )
        ],
    )

    result = transcribe_file(
        TranscribeRequest(
            input_path=input_path,
            output_dir=output_dir,
            language="ko",
            config=build_app_config(),
        )
    )

    assert result.status == "partial"
    assert result.output_path.exists()
    assert result.audio_path.exists()
    assert result.run_path.exists()
    assert result.segments_path.exists()
    run_payload = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run_payload["status"] == "partial"
    assert run_payload["phase"] == "phase1-audio-extract"
    assert run_payload["vad_backend"] == "firered"
    assert run_payload["vad_segments"] == 1
    assert run_payload["asr_model_id"].startswith("Qwen/")
    assert "테스트 문장입니다." in result.output_path.read_text(
        encoding="utf-8"
    )


def test_batch_transcribe_returns_failed_for_empty_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_dir = tmp_path / "outputs"

    result = batch_transcribe(
        BatchRequest(
            input_dir=input_dir,
            output_dir=output_dir,
            glob_pattern="*.mp4",
            language=None,
            config=build_app_config(),
        )
    )

    assert result.status == "failed"
    assert result.total == 0
