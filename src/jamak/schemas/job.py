from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscribeJob:
    input_path: Path
    output_dir: Path
    language: str | None = None


@dataclass(frozen=True)
class BatchJob:
    input_dir: Path
    output_dir: Path
    glob_pattern: str = "*.*"
    language: str | None = None

