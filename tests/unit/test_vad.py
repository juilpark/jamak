from __future__ import annotations

from pathlib import Path

from jamak.core.vad import _resolve_firered_model_dir, detect_speech_segments
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

    def fake_detect(*, audio_path: Path, model_dir: Path | None, use_gpu: bool) -> list[SpeechSegment]:
        assert audio_path.name == "dummy.wav"
        assert model_dir is None
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

    def raise_detect(*, audio_path: Path, model_dir: Path | None, use_gpu: bool) -> list[SpeechSegment]:
        assert audio_path.name == "dummy.wav"
        assert model_dir is None
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


def test_vad_model_path_uses_existing_files(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "VAD"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pth.tar").write_bytes(b"x")
    (model_dir / "cmvn.ark").write_text("cmvn", encoding="utf-8")

    called = {"count": 0}

    def fake_snapshot_download(**_: object) -> str:
        called["count"] += 1
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    resolved = _resolve_firered_model_dir(model_dir)
    assert resolved == model_dir
    assert called["count"] == 0


def test_vad_model_path_downloads_when_missing(tmp_path: Path, monkeypatch) -> None:
    root_dir = tmp_path / "FireRedVAD"
    model_dir = root_dir / "VAD"
    def fake_snapshot_download(**kwargs: object) -> str:
        local_dir = Path(str(kwargs["local_dir"]))
        vad_dir = local_dir / "VAD"
        vad_dir.mkdir(parents=True, exist_ok=True)
        (vad_dir / "model.pth.tar").write_bytes(b"x")
        (vad_dir / "cmvn.ark").write_text("cmvn", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    resolved = _resolve_firered_model_dir(model_dir)
    assert resolved == model_dir
    assert (model_dir / "model.pth.tar").exists()


def test_vad_default_model_path_uses_hf_default_cache(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot_root = tmp_path / "hf-snapshot"
    vad_dir = snapshot_root / "VAD"
    called_kwargs: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        called_kwargs.update(kwargs)
        vad_dir.mkdir(parents=True, exist_ok=True)
        (vad_dir / "model.pth.tar").write_bytes(b"x")
        (vad_dir / "cmvn.ark").write_text("cmvn", encoding="utf-8")
        return str(snapshot_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    resolved = _resolve_firered_model_dir(None)
    assert resolved == vad_dir
    assert "local_dir" not in called_kwargs
