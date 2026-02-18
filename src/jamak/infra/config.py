from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

PYTHON_POLICY_CHAIN: tuple[tuple[int, int], ...] = ((3, 14), (3, 13), (3, 12))
MVP_OUTPUT_FORMAT = "srt"
SUPPORTED_OUTPUT_FORMATS = {MVP_OUTPUT_FORMAT}
SUPPORTED_VAD_BACKENDS = {"auto", "firered", "fallback"}
DEFAULT_FIRERED_VAD_MODEL_DIR = Path("pretrained_models/FireRedVAD/VAD")


@dataclass(frozen=True)
class PythonPolicyEvaluation:
    current: tuple[int, int]
    status: str
    message: str


@dataclass(frozen=True)
class AppConfig:
    device: str
    hf_cache: Path
    output_format: str
    vad_backend: str
    vad_model_dir: Path
    python_policy: PythonPolicyEvaluation


def resolve_hf_cache_dir(custom_path: Path | None = None) -> Path:
    if custom_path is not None:
        return custom_path.expanduser().resolve()
    env_path = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface").resolve()


def evaluate_python_policy(version: tuple[int, int] | None = None) -> PythonPolicyEvaluation:
    current = version or (sys.version_info.major, sys.version_info.minor)
    highest = PYTHON_POLICY_CHAIN[0]
    lowest = PYTHON_POLICY_CHAIN[-1]
    if current >= highest:
        return PythonPolicyEvaluation(
            current=current,
            status="preferred",
            message=f"Python {current[0]}.{current[1]} is on or above preferred target {highest[0]}.{highest[1]}.",
        )
    if current in PYTHON_POLICY_CHAIN[1:]:
        return PythonPolicyEvaluation(
            current=current,
            status="fallback",
            message=(
                f"Python {current[0]}.{current[1]} is allowed by fallback policy "
                f"(priority: 3.14 -> 3.13 -> 3.12)."
            ),
        )
    if current < lowest:
        return PythonPolicyEvaluation(
            current=current,
            status="unsupported",
            message=(
                f"Python {current[0]}.{current[1]} is below minimum policy "
                f"{lowest[0]}.{lowest[1]}. Upgrade to 3.12+."
            ),
        )
    return PythonPolicyEvaluation(
        current=current,
        status="custom",
        message=(
            f"Python {current[0]}.{current[1]} is not in explicit policy chain "
            f"(3.14 -> 3.13 -> 3.12). Verify dependency compatibility."
        ),
    )


def normalize_device(value: str) -> str:
    device = value.strip().lower()
    allowed = {"auto", "cpu", "cuda", "rocm", "intel", "mps"}
    if device not in allowed:
        raise ValueError(f"Unsupported device '{value}'. Allowed: {', '.join(sorted(allowed))}")
    return device


def normalize_output_format(value: str) -> str:
    fmt = value.strip().lower()
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format '{value}'. Allowed: {sorted(SUPPORTED_OUTPUT_FORMATS)}")
    return fmt


def normalize_vad_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend not in SUPPORTED_VAD_BACKENDS:
        raise ValueError(
            f"Unsupported VAD backend '{value}'. Allowed: {sorted(SUPPORTED_VAD_BACKENDS)}"
        )
    return backend


def resolve_vad_model_dir(custom_path: Path | None = None) -> Path:
    if custom_path is not None:
        return custom_path.expanduser().resolve()
    env_path = os.getenv("JAMAK_FIRERED_VAD_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_FIRERED_VAD_MODEL_DIR.resolve()


def build_app_config(
    *,
    device: str = "auto",
    hf_cache: Path | None = None,
    output_format: str = MVP_OUTPUT_FORMAT,
    vad_backend: str = "auto",
    vad_model_dir: Path | None = None,
) -> AppConfig:
    return AppConfig(
        device=normalize_device(device),
        hf_cache=resolve_hf_cache_dir(hf_cache),
        output_format=normalize_output_format(output_format),
        vad_backend=normalize_vad_backend(vad_backend),
        vad_model_dir=resolve_vad_model_dir(vad_model_dir),
        python_policy=evaluate_python_policy(),
    )
