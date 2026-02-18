from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from jamak.schemas.subtitle import SubtitleCue

DEFAULT_TRANSLATION_PROVIDER = "openrouter"
DEFAULT_TRANSLATION_MODEL = "openai/gpt-4o-mini"
SUPPORTED_TRANSLATION_PROVIDERS: tuple[str, ...] = (DEFAULT_TRANSLATION_PROVIDER,)
TRANSLATION_SYSTEM_PROMPT = (
    "Role: Expert Subtitle Localizer & Translator\n\n"
    "Instructions:\n"
    "1. Translate the provided text into natural, idiomatic Korean.\n"
    "2. Maintain the tone and nuances of the original dialogue (e.g., informal vs. formal) based on the context.\n"
    "3. Keep the output concise and easy to read, as it is intended for subtitles.\n"
    "4. Strictly preserve all original line breaks (\\n).\n"
    "5. Output ONLY the translated text. Do not include any introductions, explanations, or quotes.\n"
    "6. If the source language is not specified, auto-detect it and translate to Korean.\n\n"
    "Target Language: Korean\n"
    "Format: Plain text only, preserving line breaks."
)


@dataclass(frozen=True)
class TranslationUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class TranslationProgress:
    completed: int
    total: int
    cue_index: int
    cue: SubtitleCue
    usage: TranslationUsage


TranslationProgressCallback = Callable[[TranslationProgress], None]


@dataclass(frozen=True)
class TranslationRequest:
    cues: list[SubtitleCue]
    target_language: str
    model: str = DEFAULT_TRANSLATION_MODEL
    provider: str = DEFAULT_TRANSLATION_PROVIDER
    source_language: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_concurrency: int = 4


class TranslationAdapter(Protocol):
    def translate(
        self,
        request: TranslationRequest,
        *,
        on_progress: TranslationProgressCallback | None = None,
    ) -> list[SubtitleCue]:
        ...


def _extract_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise RuntimeError("Translation provider returned no choices.")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None and isinstance(first_choice, dict):
        message = first_choice.get("message")
    if message is None:
        raise RuntimeError("Translation provider returned no message.")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    parts.append(part.strip())
                continue
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    raise RuntimeError("Translation provider returned unsupported message content.")


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_usage(response: Any) -> TranslationUsage:
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None and isinstance(response, dict):
        usage_obj = response.get("usage")
    usage = _to_mapping(usage_obj)
    prompt_tokens = _to_int(usage.get("prompt_tokens"))
    completion_tokens = _to_int(usage.get("completion_tokens"))
    total_tokens = _to_int(usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    cost_usd = _to_float(usage.get("cost"))
    return TranslationUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


def _create_openrouter_client(api_key: str) -> Any:
    try:
        from openrouter import OpenRouter
    except ImportError as exc:
        raise RuntimeError(
            "openrouter package is not installed. Run `uv add openrouter`."
        ) from exc
    return OpenRouter(api_key=api_key)


def _send_openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> Any:
    client = _create_openrouter_client(api_key)
    try:
        return client.chat.send(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    finally:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            close_fn()


def _build_translation_messages(
    *,
    source_language: str | None,
    target_language: str,
) -> list[dict[str, str]]:
    del source_language, target_language
    return [{"role": "system", "content": TRANSLATION_SYSTEM_PROMPT}]


def _build_translation_request_messages(
    *,
    text: str,
    source_language: str | None,
    target_language: str,
) -> list[dict[str, str]]:
    messages = _build_translation_messages(
        source_language=source_language,
        target_language=target_language,
    )
    messages.append({"role": "user", "content": text})
    return messages


class OpenRouterTranslationAdapter:
    def _translate_single_cue(
        self,
        *,
        cue: SubtitleCue,
        request: TranslationRequest,
        api_key: str,
    ) -> tuple[SubtitleCue, TranslationUsage]:
        source_text = cue.text.strip()
        if not source_text:
            return cue, TranslationUsage()
        response = _send_openrouter_chat(
            api_key=api_key,
            model=request.model,
            messages=_build_translation_request_messages(
                text=source_text,
                source_language=request.source_language,
                target_language=request.target_language,
            ),
            temperature=request.temperature,
        )
        target_text = _extract_message_content(response)
        if not target_text:
            target_text = source_text
        return (
            SubtitleCue(start=cue.start, end=cue.end, text=target_text),
            _extract_usage(response),
        )

    def translate(
        self,
        request: TranslationRequest,
        *,
        on_progress: TranslationProgressCallback | None = None,
    ) -> list[SubtitleCue]:
        api_key = request.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenRouter API key is missing. Set OPENROUTER_API_KEY or pass --api-key."
            )
        if not request.target_language.strip():
            raise ValueError("target_language must not be empty.")
        if request.max_concurrency <= 0:
            raise ValueError(
                f"max_concurrency must be > 0, got {request.max_concurrency}"
            )

        translated: list[SubtitleCue | None] = [None] * len(request.cues)
        translate_targets: list[tuple[int, SubtitleCue]] = []
        for index, cue in enumerate(request.cues):
            if cue.text.strip():
                translate_targets.append((index, cue))
            else:
                translated[index] = cue
        if not translate_targets:
            return [cue for cue in request.cues]

        worker_count = min(request.max_concurrency, len(translate_targets))
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    self._translate_single_cue,
                    cue=cue,
                    request=request,
                    api_key=api_key,
                ): index
                for index, cue in translate_targets
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    translated_cue, usage = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Translation request failed at cue index {index}: {exc}"
                    ) from exc
                translated[index] = translated_cue
                completed += 1
                if on_progress is not None:
                    try:
                        on_progress(
                            TranslationProgress(
                                completed=completed,
                                total=len(translate_targets),
                                cue_index=index,
                                cue=translated_cue,
                                usage=usage,
                            )
                        )
                    except Exception:
                        # Keep translation running even if caller-side logging fails.
                        pass

        if any(cue is None for cue in translated):
            raise RuntimeError("Translation produced incomplete cue results.")
        return [cue for cue in translated if cue is not None]


def normalize_translation_provider(provider: str) -> str:
    value = provider.strip().lower()
    if value not in SUPPORTED_TRANSLATION_PROVIDERS:
        supported = ", ".join(SUPPORTED_TRANSLATION_PROVIDERS)
        raise ValueError(
            f"Unsupported translation provider '{provider}'. Supported: {supported}"
        )
    return value


def translate_subtitles(
    request: TranslationRequest,
    *,
    on_progress: TranslationProgressCallback | None = None,
) -> list[SubtitleCue]:
    provider = normalize_translation_provider(request.provider)
    adapters: dict[str, TranslationAdapter] = {
        DEFAULT_TRANSLATION_PROVIDER: OpenRouterTranslationAdapter(),
    }
    adapter = adapters[provider]
    return adapter.translate(request, on_progress=on_progress)


def translate_with_openai_compatible(request: TranslationRequest) -> list[SubtitleCue]:
    """Backward-compatible alias for provider-based translation."""
    return translate_subtitles(request)
