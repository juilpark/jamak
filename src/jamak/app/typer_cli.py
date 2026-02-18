from __future__ import annotations

from pathlib import Path

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from jamak.core.pipeline import (
    BatchRequest,
    TranscribeRequest,
    batch_transcribe,
    transcribe_file,
)
from jamak.core.subtitle import read_srt, write_srt
from jamak.core.translate import (
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TRANSLATION_PROVIDER,
    TranslationProgress,
    TranslationRequest,
    normalize_translation_provider,
    translate_subtitles,
)
from jamak.infra.config import build_app_config
from jamak.infra.doctor import collect_doctor_report, render_doctor_report
from jamak.infra.language import normalize_and_validate_language
from jamak.infra.storage import ensure_directory

app = typer.Typer(
    name="jamak",
    add_completion=False,
    help="CLI-first subtitle generation pipeline.",
)


def _validate_language_option(language: str | None) -> str | None:
    if language is None:
        return None
    try:
        return normalize_and_validate_language(language)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--language") from exc


def _build_translated_output_path(input_srt: Path, target_language: str) -> Path:
    slug = (
        target_language.strip().lower().replace(" ", "-").replace("/", "-")
        or "translated"
    )
    return input_srt.with_name(f"{input_srt.stem}.{slug}.srt")


@app.command("transcribe")
def transcribe_command(
    input_path: Path = typer.Argument(..., help="Input audio/video file path."),
    output_dir: Path = typer.Option(
        Path("./outputs"), "--output-dir", "-o", help="Directory for output files."
    ),
    language: str | None = typer.Option(
        None, "--language", help="Forced language name (e.g. Japanese)."
    ),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda (MVP)"),
    vad_backend: str = typer.Option(
        "auto", "--vad-backend", help="auto|firered|fallback"
    ),
    vad_model_dir: Path | None = typer.Option(
        None, "--vad-model-dir", help="Path to FireRedVAD model directory."
    ),
    asr_model: str | None = typer.Option(
        None, "--asr-model", help="ASR model id (default: Qwen/Qwen3-ASR-1.7B)."
    ),
    asr_max_new_tokens: int = typer.Option(
        256, "--asr-max-new-tokens", help="Maximum ASR generation tokens per segment."
    ),
    asr_batch_size: int = typer.Option(
        8, "--asr-batch-size", help="ASR inference batch size."
    ),
    align_model: str | None = typer.Option(
        None, "--align-model", help="Forced aligner model id (default: Qwen/Qwen3-ForcedAligner-0.6B)."
    ),
    align_batch_size: int = typer.Option(
        8, "--align-batch-size", help="Forced aligner inference batch size."
    ),
    hf_cache: Path | None = typer.Option(
        None, "--hf-cache", help="Custom Hugging Face cache path."
    ),
) -> None:
    """Transcribe a single file."""
    if not input_path.exists():
        raise typer.BadParameter(f"Input not found: {input_path}")
    language = _validate_language_option(language)

    config = build_app_config(
        device=device,
        hf_cache=hf_cache,
        output_format="srt",
        vad_backend=vad_backend,
        vad_model_dir=vad_model_dir,
        asr_model_id=asr_model,
        asr_max_new_tokens=asr_max_new_tokens,
        asr_batch_size=asr_batch_size,
        align_model_id=align_model,
        align_batch_size=align_batch_size,
    )
    request = TranscribeRequest(
        input_path=input_path,
        output_dir=output_dir,
        language=language,
        config=config,
    )
    result = transcribe_file(request)
    typer.echo(
        f"[{result.status}] {result.message}\n"
        f"- input: {result.input_path}\n"
        f"- output srt: {result.output_path}\n"
        f"- extracted audio: {result.audio_path}\n"
        f"- run metadata: {result.run_path}\n"
        f"- segments metadata: {result.segments_path}\n"
        f"- run log: {result.log_path}"
    )
    if result.status == "failed":
        raise typer.Exit(code=2)


@app.command("translate")
def translate_command(
    input_srt: Path = typer.Argument(..., help="Input subtitle SRT file path."),
    target_language: str = typer.Option(
        ...,
        "--target-language",
        "-t",
        help="Target language name (example: Korean, English, Japanese).",
    ),
    output_path: Path | None = typer.Option(
        None, "--output-path", "-o", help="Translated SRT output path."
    ),
    provider: str = typer.Option(
        DEFAULT_TRANSLATION_PROVIDER,
        "--provider",
        help="Translation provider adapter. Default: openrouter.",
    ),
    model: str = typer.Option(
        DEFAULT_TRANSLATION_MODEL,
        "--model",
        help="Provider model id (example: openai/gpt-4o-mini).",
    ),
    source_language: str | None = typer.Option(
        None, "--source-language", help="Optional source language hint."
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        "--openrouter-api-key",
        "-k",
        help="Provider API key (default: OPENROUTER_API_KEY environment variable).",
    ),
    temperature: float = typer.Option(
        0.0, "--temperature", help="Translation sampling temperature."
    ),
    max_concurrency: int = typer.Option(
        4, "--max-concurrency", help="Max parallel translation requests."
    ),
) -> None:
    """Translate an existing SRT file using provider adapters."""
    if not input_srt.exists() or not input_srt.is_file():
        raise typer.BadParameter(f"Input SRT not found: {input_srt}")
    if input_srt.suffix.lower() != ".srt":
        raise typer.BadParameter(
            f"Only .srt input is supported for translation, got: {input_srt}"
        )
    if temperature < 0.0 or temperature > 2.0:
        raise typer.BadParameter(
            f"temperature must be between 0.0 and 2.0, got {temperature}"
        )
    if max_concurrency <= 0:
        raise typer.BadParameter(
            f"max_concurrency must be > 0, got {max_concurrency}",
            param_hint="--max-concurrency",
        )
    try:
        provider = normalize_translation_provider(provider)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--provider") from exc

    cues = read_srt(input_srt)
    if not cues:
        raise typer.BadParameter(
            f"No subtitle cues found in SRT file: {input_srt}"
        )
    translatable_count = sum(1 for cue in cues if cue.text.strip())

    request = TranslationRequest(
        cues=cues,
        target_language=target_language,
        model=model,
        provider=provider,
        source_language=source_language,
        api_key=api_key,
        temperature=temperature,
        max_concurrency=max_concurrency,
    )

    translated_cues = []
    if translatable_count > 0:
        translation_error: Exception | None = None
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]cost: {task.fields[cost]}"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                description="Translating subtitles...",
                total=translatable_count,
                cost="$0.000000",
            )
            cumulative_cost_usd = 0.0

            def _progress_log(
                event: TranslationProgress,
            ) -> None:
                nonlocal cumulative_cost_usd
                if event.usage.cost_usd is not None:
                    cumulative_cost_usd += event.usage.cost_usd
                progress.update(
                    task_id,
                    advance=1,
                    cost=f"${cumulative_cost_usd:.6f}",
                )

            try:
                translated_cues = translate_subtitles(
                    request,
                    on_progress=_progress_log,
                )
            except Exception as exc:
                translation_error = exc
        if translation_error is not None:
            typer.echo(f"[failed] Subtitle translation failed: {translation_error}")
            raise typer.Exit(code=2) from translation_error
    else:
        try:
            translated_cues = translate_subtitles(request)
        except Exception as exc:
            typer.echo(f"[failed] Subtitle translation failed: {exc}")
            raise typer.Exit(code=2) from exc

    resolved_output_path = output_path or _build_translated_output_path(
        input_srt, target_language
    )
    ensure_directory(resolved_output_path.parent)
    write_srt(translated_cues, resolved_output_path)
    typer.echo(
        "[done] Subtitle translation complete.\n"
        f"- input: {input_srt}\n"
        f"- output: {resolved_output_path}\n"
        f"- provider: {provider}\n"
        f"- model: {model}\n"
        f"- max concurrency: {max_concurrency}\n"
        f"- cues: {len(translated_cues)}"
    )


@app.command("batch")
def batch_command(
    input_dir: Path = typer.Argument(..., help="Directory containing input files."),
    output_dir: Path = typer.Option(
        Path("./outputs"), "--output-dir", "-o", help="Directory for output files."
    ),
    glob_pattern: str = typer.Option(
        "*.*", "--glob", help="Glob pattern to select inputs (default: *.*)."
    ),
    language: str | None = typer.Option(
        None, "--language", help="Forced language name (e.g. Japanese)."
    ),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda (MVP)"),
    vad_backend: str = typer.Option(
        "auto", "--vad-backend", help="auto|firered|fallback"
    ),
    vad_model_dir: Path | None = typer.Option(
        None, "--vad-model-dir", help="Path to FireRedVAD model directory."
    ),
    asr_model: str | None = typer.Option(
        None, "--asr-model", help="ASR model id (default: Qwen/Qwen3-ASR-1.7B)."
    ),
    asr_max_new_tokens: int = typer.Option(
        256, "--asr-max-new-tokens", help="Maximum ASR generation tokens per segment."
    ),
    asr_batch_size: int = typer.Option(
        8, "--asr-batch-size", help="ASR inference batch size."
    ),
    align_model: str | None = typer.Option(
        None, "--align-model", help="Forced aligner model id (default: Qwen/Qwen3-ForcedAligner-0.6B)."
    ),
    align_batch_size: int = typer.Option(
        8, "--align-batch-size", help="Forced aligner inference batch size."
    ),
    hf_cache: Path | None = typer.Option(
        None, "--hf-cache", help="Custom Hugging Face cache path."
    ),
) -> None:
    """Run batch transcription."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise typer.BadParameter(f"Input directory not found: {input_dir}")
    language = _validate_language_option(language)

    config = build_app_config(
        device=device,
        hf_cache=hf_cache,
        output_format="srt",
        vad_backend=vad_backend,
        vad_model_dir=vad_model_dir,
        asr_model_id=asr_model,
        asr_max_new_tokens=asr_max_new_tokens,
        asr_batch_size=asr_batch_size,
        align_model_id=align_model,
        align_batch_size=align_batch_size,
    )
    request = BatchRequest(
        input_dir=input_dir,
        output_dir=output_dir,
        glob_pattern=glob_pattern,
        language=language,
        config=config,
    )
    result = batch_transcribe(request)
    typer.echo(
        f"[{result.status}] {result.message}\n"
        f"- files discovered: {result.total}\n"
        f"- succeeded: {result.succeeded}\n"
        f"- partial: {result.partial}\n"
        f"- failed: {result.failed}"
    )
    if result.status == "failed":
        raise typer.Exit(code=2)


@app.command("doctor")
def doctor_command(
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda (MVP)"),
    vad_backend: str = typer.Option(
        "auto", "--vad-backend", help="auto|firered|fallback"
    ),
    vad_model_dir: Path | None = typer.Option(
        None, "--vad-model-dir", help="Path to FireRedVAD model directory."
    ),
    asr_model: str | None = typer.Option(
        None, "--asr-model", help="ASR model id (default: Qwen/Qwen3-ASR-1.7B)."
    ),
    asr_max_new_tokens: int = typer.Option(
        256, "--asr-max-new-tokens", help="Maximum ASR generation tokens per segment."
    ),
    asr_batch_size: int = typer.Option(
        8, "--asr-batch-size", help="ASR inference batch size."
    ),
    align_model: str | None = typer.Option(
        None, "--align-model", help="Forced aligner model id (default: Qwen/Qwen3-ForcedAligner-0.6B)."
    ),
    align_batch_size: int = typer.Option(
        8, "--align-batch-size", help="Forced aligner inference batch size."
    ),
    hf_cache: Path | None = typer.Option(
        None, "--hf-cache", help="Custom Hugging Face cache path."
    ),
) -> None:
    """Check runtime readiness (Python/ffmpeg/cache/device)."""
    config = build_app_config(
        device=device,
        hf_cache=hf_cache,
        output_format="srt",
        vad_backend=vad_backend,
        vad_model_dir=vad_model_dir,
        asr_model_id=asr_model,
        asr_max_new_tokens=asr_max_new_tokens,
        asr_batch_size=asr_batch_size,
        align_model_id=align_model,
        align_batch_size=align_batch_size,
    )
    report = collect_doctor_report(config)
    typer.echo(render_doctor_report(report))
    if not report.ok:
        raise typer.Exit(code=1)


def run() -> None:
    """Console-script entrypoint."""
    app()
