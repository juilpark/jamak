from __future__ import annotations

from pathlib import Path
from typing import Protocol

from jamak.schemas.segment import SpeechSegment
from jamak.schemas.subtitle import SubtitleCue


class AlignmentBackend(Protocol):
    """Interface for forced alignment engines such as Qwen3-ForcedAligner."""

    def align(
        self, audio_path: Path, segments: list[SpeechSegment], texts: list[str]
    ) -> list[SubtitleCue]:
        """Return subtitle cues with aligned timestamps."""

