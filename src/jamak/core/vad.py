from __future__ import annotations

from pathlib import Path
from typing import Protocol

from jamak.schemas.segment import SpeechSegment


class VADBackend(Protocol):
    """Interface for VAD engines such as FireRedVAD."""

    def detect(self, audio_path: Path) -> list[SpeechSegment]:
        """Return speech segments with start/end timestamps."""

