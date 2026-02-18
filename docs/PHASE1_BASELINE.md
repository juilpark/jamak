# Phase 1 Baseline (RTF)

- 측정일: 2026-02-18
- 환경: Apple Silicon Mac (local), Python 3.13, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- 모델:
  - ASR: `Qwen/Qwen3-ASR-0.6B`
  - Forced Aligner: `Qwen/Qwen3-ForcedAligner-0.6B`

## 측정 명령

```bash
/usr/bin/time -p sh -c 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python main.py transcribe test_audio.mp3 -o .tmp/bench --vad-backend fallback --asr-model Qwen/Qwen3-ASR-0.6B --asr-max-new-tokens 64 --asr-batch-size 1 --align-model Qwen/Qwen3-ForcedAligner-0.6B --align-batch-size 1'

/usr/bin/time -p sh -c 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python main.py transcribe test_video_short.mp4 -o .tmp/bench --vad-backend firered --asr-model Qwen/Qwen3-ASR-0.6B --asr-max-new-tokens 64 --asr-batch-size 8 --align-model Qwen/Qwen3-ForcedAligner-0.6B --align-batch-size 8'
```

## 결과

| Sample | Audio Duration (s) | Wall Time (s) | RTF (wall / duration) |
|---|---:|---:|---:|
| `test_audio.mp3` | 257.354 | 88.08 | 0.3423 |
| `test_video_short.mp4` | 288.045 | 188.50 | 0.6544 |

## 해석

- 두 샘플 모두 `RTF < 1.0`로 실시간보다 빠르게 처리됨.
- 짧고 깨끗한 단일 화자 음성(`test_audio.mp3`)이 더 빠름.
- 화자/언어 혼합과 세그먼트 수가 많은 영상(`test_video_short.mp4`)에서 처리 비용 증가.

## 남은 항목

- CER/WER 측정은 정답 스크립트(ground truth) 준비 후 수행.
