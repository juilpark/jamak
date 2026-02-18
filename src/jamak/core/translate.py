from __future__ import annotations

from dataclasses import dataclass

from jamak.schemas.subtitle import SubtitleCue


@dataclass(frozen=True)
class TranslationRequest:
    cues: list[SubtitleCue]
    target_language: str
    model: str


def translate_with_openai_compatible(_: TranslationRequest) -> list[SubtitleCue]:
    """Placeholder for Phase 4 translation implementation."""
    raise NotImplementedError("Translation is planned for Phase 4.")

