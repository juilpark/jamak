from __future__ import annotations

from pathlib import Path

from jamak.core.vad import detect_speech_segments
from jamak.infra.config import build_app_config
from jamak.schemas.segment import SpeechSegment


def test_vad_fallback_backend_returns_full_range_segment() -> None:
    config = build_app_config(vad_backend="fallback")
    result = detect_speech_segments(
        Path("dummy.wav"),
        config=config,
        duration_seconds=12.3,
    )
    assert result.backend == "fallback"
    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 12.3


def test_vad_auto_uses_firered_when_detection_succeeds(monkeypatch) -> None:
    config = build_app_config(vad_backend="auto")

    def fake_detect(*, audio_path: Path, model_dir: Path, use_gpu: bool) -> list[SpeechSegment]:
        assert audio_path.name == "dummy.wav"
        assert model_dir
        assert use_gpu is False
        return [SpeechSegment(start=1.0, end=2.0, confidence=0.9)]

    monkeypatch.setattr("jamak.core.vad._detect_with_firered", fake_detect)
    result = detect_speech_segments(
        Path("dummy.wav"),
        config=config,
        duration_seconds=5.0,
    )
    assert result.backend == "firered"
    assert len(result.segments) == 1
    assert result.segments[0].start == 1.0


def test_vad_auto_falls_back_when_firered_fails(monkeypatch) -> None:
    config = build_app_config(vad_backend="auto")

    def raise_detect(*, audio_path: Path, model_dir: Path, use_gpu: bool) -> list[SpeechSegment]:
        assert audio_path.name == "dummy.wav"
        assert model_dir
        assert use_gpu is False
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("jamak.core.vad._detect_with_firered", raise_detect)
    result = detect_speech_segments(
        Path("dummy.wav"),
        config=config,
        duration_seconds=8.0,
    )
    assert result.backend == "fallback"
    assert len(result.segments) == 1
    assert "unavailable" in result.message
