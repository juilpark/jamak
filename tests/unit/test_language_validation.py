from __future__ import annotations

import pytest

from jamak.infra.language import normalize_and_validate_language


def test_language_validation_accepts_supported_name_case_insensitive() -> None:
    assert normalize_and_validate_language("japanese") == "Japanese"


def test_language_validation_rejects_language_code() -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        normalize_and_validate_language("ja")
