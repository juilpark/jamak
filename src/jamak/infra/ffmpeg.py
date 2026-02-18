from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def get_ffmpeg_path() -> Path | None:
    path = shutil.which("ffmpeg")
    return Path(path) if path else None


def get_ffprobe_path() -> Path | None:
    path = shutil.which("ffprobe")
    return Path(path) if path else None


def get_ffmpeg_version() -> str | None:
    ffmpeg = get_ffmpeg_path()
    if ffmpeg is None:
        return None
    proc = subprocess.run(
        [str(ffmpeg), "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout.splitlines()[0].strip()


def extract_audio(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> None:
    ffmpeg = get_ffmpeg_path()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"Failed to extract audio: {detail}")


def probe_duration_seconds(input_path: Path) -> float | None:
    ffprobe = get_ffprobe_path()
    if ffprobe is None:
        return None
    proc = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    raw = proc.stdout.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
