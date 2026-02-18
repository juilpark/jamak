from __future__ import annotations

from pathlib import Path
from typing import Protocol

from jamak.schemas.segment import SpeechSegment


class ASRBackend(Protocol):
    """Interface for ASR engines such as Qwen3-ASR."""

    def transcribe(self, audio_path: Path, segments: list[SpeechSegment]) -> list[str]:
        """Return text for each speech segment."""

