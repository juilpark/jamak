from __future__ import annotations

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
)

_SUPPORTED_LANGUAGE_SET = frozenset(SUPPORTED_LANGUAGES)


def normalize_language_name(language: str) -> str:
    value = language.strip()
    return value[:1].upper() + value[1:].lower() if value else value


def normalize_and_validate_language(language: str) -> str:
    normalized = normalize_language_name(language)
    if not normalized:
        raise ValueError("Language must not be empty.")
    if normalized not in _SUPPORTED_LANGUAGE_SET:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(
            f"Unsupported language '{language}'. Use one of: {supported}"
        )
    return normalized
