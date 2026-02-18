from __future__ import annotations

from pathlib import Path

import pytest

from jamak.core.subtitle import read_srt, write_srt
from jamak.schemas.subtitle import SubtitleCue


def test_srt_read_write_roundtrip(tmp_path: Path) -> None:
    input_cues = [
        SubtitleCue(start=0.0, end=1.234, text="첫 번째 줄"),
        SubtitleCue(start=2.5, end=4.0, text="둘째 줄\n줄바꿈"),
    ]
    output_path = tmp_path / "sample.srt"

    write_srt(input_cues, output_path)
    loaded = read_srt(output_path)

    assert len(loaded) == 2
    assert loaded[0].start == pytest.approx(0.0)
    assert loaded[0].end == pytest.approx(1.234, abs=0.001)
    assert loaded[0].text == "첫 번째 줄"
    assert loaded[1].start == pytest.approx(2.5)
    assert loaded[1].end == pytest.approx(4.0)
    assert loaded[1].text == "둘째 줄\n줄바꿈"
