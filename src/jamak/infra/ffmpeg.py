from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def get_ffmpeg_path() -> Path | None:
    path = shutil.which("ffmpeg")
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


def extract_audio(_: Path, __: Path) -> None:
    """Placeholder for audio extraction wiring in Phase 1."""
    raise NotImplementedError("Audio extraction is planned for Phase 1.")

