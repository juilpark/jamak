from __future__ import annotations

from pathlib import Path

import typer

from jamak.core.pipeline import (
    BatchRequest,
    TranscribeRequest,
    batch_transcribe,
    transcribe_file,
)
from jamak.infra.config import build_app_config
from jamak.infra.doctor import collect_doctor_report, render_doctor_report

app = typer.Typer(
    name="jamak",
    add_completion=False,
    help="CLI-first subtitle generation pipeline.",
)


@app.command("transcribe")
def transcribe_command(
    input_path: Path = typer.Argument(..., help="Input audio/video file path."),
    output_dir: Path = typer.Option(
        Path("./outputs"), "--output-dir", "-o", help="Directory for output files."
    ),
    language: str | None = typer.Option(None, "--language", help="Language code hint."),
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
    hf_cache: Path | None = typer.Option(
        None, "--hf-cache", help="Custom Hugging Face cache path."
    ),
) -> None:
    """Transcribe a single file."""
    if not input_path.exists():
        raise typer.BadParameter(f"Input not found: {input_path}")

    config = build_app_config(
        device=device,
        hf_cache=hf_cache,
        output_format="srt",
        vad_backend=vad_backend,
        vad_model_dir=vad_model_dir,
        asr_model_id=asr_model,
        asr_max_new_tokens=asr_max_new_tokens,
        asr_batch_size=asr_batch_size,
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
        f"- segments metadata: {result.segments_path}"
    )
    if result.status == "failed":
        raise typer.Exit(code=2)


@app.command("batch")
def batch_command(
    input_dir: Path = typer.Argument(..., help="Directory containing input files."),
    output_dir: Path = typer.Option(
        Path("./outputs"), "--output-dir", "-o", help="Directory for output files."
    ),
    glob_pattern: str = typer.Option(
        "*.*", "--glob", help="Glob pattern to select inputs (default: *.*)."
    ),
    language: str | None = typer.Option(None, "--language", help="Language code hint."),
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
    hf_cache: Path | None = typer.Option(
        None, "--hf-cache", help="Custom Hugging Face cache path."
    ),
) -> None:
    """Run batch transcription."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise typer.BadParameter(f"Input directory not found: {input_dir}")

    config = build_app_config(
        device=device,
        hf_cache=hf_cache,
        output_format="srt",
        vad_backend=vad_backend,
        vad_model_dir=vad_model_dir,
        asr_model_id=asr_model,
        asr_max_new_tokens=asr_max_new_tokens,
        asr_batch_size=asr_batch_size,
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
    )
    report = collect_doctor_report(config)
    typer.echo(render_doctor_report(report))
    if not report.ok:
        raise typer.Exit(code=1)


def run() -> None:
    """Console-script entrypoint."""
    app()
