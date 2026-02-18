# jamak

CLI-first subtitle generation pipeline.

Current Phase 1 pipeline:

1. audio extract (`ffmpeg`)
2. VAD (`FireRedVAD`)
3. ASR (`Qwen3-ASR`)
4. forced alignment (`Qwen3-ForcedAligner`)
5. SRT output + metadata/log artifacts

Phase 4 translation (adapter-based):

1. provider adapter interface
2. OpenRouter adapter (first provider)
3. SRT -> translated SRT CLI flow

## Requirements

- Python `3.13` (policy: `3.14 -> 3.13 -> 3.12`)
- `uv`
- `ffmpeg`

## Install

```bash
uv sync
```

## CLI

### Doctor

```bash
./.venv/bin/python main.py doctor
```

### Single file transcribe

```bash
./.venv/bin/python main.py transcribe test_audio.mp3 -o outputs \
  --vad-backend firered \
  --asr-model Qwen/Qwen3-ASR-0.6B \
  --align-model Qwen/Qwen3-ForcedAligner-0.6B
```

### Batch transcribe

```bash
./.venv/bin/python main.py batch . -o outputs --glob '*.mp3'
```

### Translate SRT (OpenRouter adapter)

```bash
OPENROUTER_API_KEY=... \
./.venv/bin/python main.py translate outputs/test_audio.srt \
  -t Korean \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --max-concurrency 4
```

### Offline reproducible run (recommended after model cache warmup)

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
./.venv/bin/python main.py transcribe test_audio.mp3 -o outputs
```

## Outputs

For input `foo.mp4`:

- SRT: `outputs/foo.srt`
- run metadata: `outputs/.jamak/foo/run.json`
- segments metadata: `outputs/.jamak/foo/segments.json`
- run log: `outputs/.jamak/foo/run.log`

## Notes

- Default cache follows Hugging Face standard cache (`HF_HOME` / `TRANSFORMERS_CACHE`).
- Network DNS issue may cause model resolution retries. Offline mode avoids that once cache is ready.
- Translation uses adapter architecture: OpenRouter is first provider, more providers can be added with new adapters.
- Phase 2+ (`FastAPI`, Docker/NAS, provider expansion) is tracked in `TODO.md`.
