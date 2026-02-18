from __future__ import annotations

from pathlib import Path
import re

from jamak.schemas.subtitle import SubtitleCue

_SRT_TIMESTAMP_PATTERN = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*$"
)


def _format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    minutes = (millis % 3_600_000) // 60_000
    secs = (millis % 60_000) // 1_000
    ms = millis % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _parse_timestamp(value: str) -> float:
    hh, mm, ss_ms = value.split(":")
    ss, ms = ss_ms.split(",")
    total_ms = (
        int(hh) * 3_600_000 + int(mm) * 60_000 + int(ss) * 1_000 + int(ms)
    )
    return total_ms / 1_000.0


def read_srt(input_path: Path) -> list[SubtitleCue]:
    content = input_path.read_text(encoding="utf-8")
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip(), flags=re.MULTILINE)
    cues: list[SubtitleCue] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        if lines[0].strip().isdigit():
            if len(lines) < 3:
                continue
            ts_line = lines[1]
            text_lines = lines[2:]
        else:
            ts_line = lines[0]
            text_lines = lines[1:]
        match = _SRT_TIMESTAMP_PATTERN.match(ts_line)
        if not match:
            continue
        start_raw, end_raw = match.group(1), match.group(2)
        text = "\n".join(text_lines).strip()
        cues.append(
            SubtitleCue(
                start=_parse_timestamp(start_raw),
                end=_parse_timestamp(end_raw),
                text=text,
            )
        )
    return cues


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
