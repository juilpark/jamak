from __future__ import annotations

import time

import pytest

from jamak.core.translate import (
    TranslationProgress,
    TranslationRequest,
    normalize_translation_provider,
    translate_subtitles,
)
from jamak.schemas.subtitle import SubtitleCue


class _FakeClient:
    class _Chat:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def send(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(kwargs)
            source_text = str(kwargs["messages"][-1]["content"])
            return {
                "choices": [
                    {"message": {"content": f"[translated]{source_text}"}}
                ]
            }

    def __init__(self) -> None:
        self.chat = self._Chat()

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_translate_subtitles_with_openrouter_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        "jamak.core.translate._create_openrouter_client",
        lambda _: _FakeClient(),
    )
    request = TranslationRequest(
        cues=[
            SubtitleCue(start=0.0, end=1.0, text="こんにちは"),
            SubtitleCue(start=1.0, end=2.0, text=""),
        ],
        target_language="Korean",
        provider="openrouter",
        model="openai/gpt-4o-mini",
        api_key="dummy",
    )

    translated = translate_subtitles(request)
    assert len(translated) == 2
    assert translated[0].start == pytest.approx(0.0)
    assert translated[0].end == pytest.approx(1.0)
    assert translated[0].text == "[translated]こんにちは"
    assert translated[1].text == ""


def test_translate_subtitles_keeps_original_order_under_parallel_requests(
    monkeypatch,
) -> None:
    def fake_send_openrouter_chat(
        *,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> dict[str, object]:
        del api_key, model, temperature
        source_text = str(messages[-1]["content"])
        if source_text == "A":
            time.sleep(0.15)
        elif source_text == "B":
            time.sleep(0.05)
        else:
            time.sleep(0.01)
        return {
            "choices": [
                {"message": {"content": f"T-{source_text}"}}
            ]
        }

    monkeypatch.setattr(
        "jamak.core.translate._send_openrouter_chat",
        fake_send_openrouter_chat,
    )
    request = TranslationRequest(
        cues=[
            SubtitleCue(start=0.0, end=1.0, text="A"),
            SubtitleCue(start=1.0, end=2.0, text="B"),
            SubtitleCue(start=2.0, end=3.0, text="C"),
        ],
        target_language="Korean",
        provider="openrouter",
        model="openai/gpt-4o-mini",
        api_key="dummy",
        max_concurrency=3,
    )

    translated = translate_subtitles(request)
    assert [cue.text for cue in translated] == ["T-A", "T-B", "T-C"]


def test_translate_subtitles_reports_progress(monkeypatch) -> None:
    def fake_send_openrouter_chat(
        *,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> dict[str, object]:
        del api_key, model, temperature
        source_text = str(messages[-1]["content"])
        return {
            "choices": [{"message": {"content": f"T-{source_text}"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": "0.000020",
            },
        }

    monkeypatch.setattr(
        "jamak.core.translate._send_openrouter_chat",
        fake_send_openrouter_chat,
    )
    progress_events: list[TranslationProgress] = []
    request = TranslationRequest(
        cues=[
            SubtitleCue(start=0.0, end=1.0, text="A"),
            SubtitleCue(start=1.0, end=2.0, text=""),
            SubtitleCue(start=2.0, end=3.0, text="C"),
        ],
        target_language="Korean",
        provider="openrouter",
        model="openai/gpt-4o-mini",
        api_key="dummy",
        max_concurrency=2,
    )

    _ = translate_subtitles(
        request,
        on_progress=lambda event: progress_events.append(event),
    )
    assert len(progress_events) == 2
    assert [event.completed for event in progress_events] == [1, 2]
    assert all(event.total == 2 for event in progress_events)
    assert {event.cue_index for event in progress_events} == {0, 2}
    assert {event.cue.text for event in progress_events} == {"T-A", "T-C"}
    assert all(event.usage.total_tokens == 15 for event in progress_events)
    assert all(event.usage.cost_usd == pytest.approx(0.000020) for event in progress_events)


def test_translation_splits_static_and_dynamic_messages(monkeypatch) -> None:
    captured_messages: list[list[dict[str, str]]] = []

    def fake_send_openrouter_chat(
        *,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> dict[str, object]:
        del api_key, model, temperature
        captured_messages.append(messages)
        source_text = str(messages[-1]["content"])
        return {"choices": [{"message": {"content": f"T-{source_text}"}}]}

    monkeypatch.setattr(
        "jamak.core.translate._send_openrouter_chat",
        fake_send_openrouter_chat,
    )
    request = TranslationRequest(
        cues=[
            SubtitleCue(start=0.0, end=1.0, text="line-a"),
            SubtitleCue(start=1.0, end=2.0, text="line-b"),
        ],
        source_language="Japanese",
        target_language="Korean",
        provider="openrouter",
        model="openai/gpt-4o-mini",
        api_key="dummy",
        max_concurrency=1,
    )

    _ = translate_subtitles(request)
    assert len(captured_messages) == 2
    assert captured_messages[0][0] == captured_messages[1][0]
    assert len(captured_messages[0]) == 2
    assert len(captured_messages[1]) == 2
    assert captured_messages[0][1]["role"] == "user"
    assert captured_messages[1][1]["role"] == "user"
    assert captured_messages[0][1]["content"] == "line-a"
    assert captured_messages[1][1]["content"] == "line-b"


def test_translate_subtitles_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    request = TranslationRequest(
        cues=[SubtitleCue(start=0.0, end=1.0, text="hello")],
        target_language="Korean",
        provider="openrouter",
        model="openai/gpt-4o-mini",
    )
    with pytest.raises(RuntimeError, match="API key"):
        translate_subtitles(request)


def test_translate_subtitles_rejects_invalid_max_concurrency() -> None:
    request = TranslationRequest(
        cues=[SubtitleCue(start=0.0, end=1.0, text="hello")],
        target_language="Korean",
        provider="openrouter",
        model="openai/gpt-4o-mini",
        api_key="dummy",
        max_concurrency=0,
    )
    with pytest.raises(ValueError, match="max_concurrency"):
        translate_subtitles(request)


def test_normalize_translation_provider_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported translation provider"):
        normalize_translation_provider("anthropic")
