from __future__ import annotations

from pathlib import Path

from jamak.schemas.subtitle import SubtitleCue


def _format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    minutes = (millis % 3_600_000) // 60_000
    secs = (millis % 60_000) // 1_000
    ms = millis % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(cues: list[SubtitleCue], output_path: Path) -> None:
    """Write subtitle cues to an SRT file."""
    lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.append(str(index))
        lines.append(
            f"{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}"
        )
        lines.append(cue.text.strip())
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")

