# Third-Party Notices

## FireRedVAD (vendored subset)

- Source: `FireRedTeam/FireRedASR2S` ([GitHub](https://github.com/FireRedTeam/FireRedASR2S))
- License: Apache-2.0
- Vendored modules:
  - `src/jamak/vendor/fireredvad/vad.py`
  - `src/jamak/vendor/fireredvad/core/audio_feat.py`
  - `src/jamak/vendor/fireredvad/core/constants.py`
  - `src/jamak/vendor/fireredvad/core/detect_model.py`
  - `src/jamak/vendor/fireredvad/core/vad_postprocessor.py`

This project includes adapted code from the files above to provide
CLI-installable FireRedVAD execution without requiring an additional repository
checkout.

## Qwen3-ASR (vendored subset)

- Source: `QwenLM/Qwen3-ASR` ([GitHub](https://github.com/QwenLM/Qwen3-ASR))
- License: Apache-2.0
- Vendored modules:
  - `src/jamak/vendor/qwen3_asr/transformers_backend/configuration_qwen3_asr.py`
  - `src/jamak/vendor/qwen3_asr/transformers_backend/modeling_qwen3_asr.py`
  - `src/jamak/vendor/qwen3_asr/transformers_backend/processing_qwen3_asr.py`
  - `src/jamak/vendor/qwen3_asr/transformers_backend/__init__.py`

These files are used to run Qwen3-ASR inference without requiring the
`qwen-asr` package install path.
