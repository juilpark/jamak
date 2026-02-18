from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float
    confidence: float | None = None

