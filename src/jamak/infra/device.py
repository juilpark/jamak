from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceReport:
    requested: str
    available: tuple[str, ...]
    selected: str
    request_satisfied: bool


def _has_command(command: str) -> bool:
    return shutil.which(command) is not None


def detect_available_devices() -> tuple[str, ...]:
    devices = ["cpu"]
    if _has_command("nvidia-smi"):
        devices.append("cuda")
    if _has_command("rocm-smi") or _has_command("rocminfo"):
        devices.append("rocm")
    if _has_command("sycl-ls") or _has_command("intel_gpu_top"):
        devices.append("intel")
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        devices.append("mps")
    return tuple(devices)


def choose_device(requested: str, available: tuple[str, ...]) -> str:
    if requested != "auto":
        return requested
    for candidate in ("cuda", "mps", "cpu"):
        if candidate in available:
            return candidate
    return "cpu"


def detect_device_report(requested: str) -> DeviceReport:
    available = detect_available_devices()
    selected = choose_device(requested, available)
    request_satisfied = requested == "auto" or requested in available
    return DeviceReport(
        requested=requested,
        available=available,
        selected=selected,
        request_satisfied=request_satisfied,
    )

