from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jamak.infra.config import AppConfig
from jamak.infra.device import detect_device_report
from jamak.infra.ffmpeg import get_ffmpeg_path, get_ffmpeg_version


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def _cache_check(path: Path) -> DoctorCheck:
    parent_exists = path.exists() or path.parent.exists()
    writable = path.exists() and path.is_dir()
    if not path.exists():
        writable = path.parent.exists()
    ok = parent_exists and writable
    return DoctorCheck(
        name="HuggingFace cache",
        ok=ok,
        detail=f"path={path}",
    )


def collect_doctor_report(config: AppConfig) -> DoctorReport:
    py_ok = config.python_policy.status in {"preferred", "fallback", "custom"}
    python_check = DoctorCheck(
        name="Python policy",
        ok=py_ok,
        detail=config.python_policy.message,
    )

    ffmpeg_path = get_ffmpeg_path()
    ffmpeg_ok = ffmpeg_path is not None
    version = get_ffmpeg_version() or "unknown"
    ffmpeg_check = DoctorCheck(
        name="ffmpeg",
        ok=ffmpeg_ok,
        detail=f"path={ffmpeg_path} version={version}",
    )

    cache_check = _cache_check(config.hf_cache)

    device_report = detect_device_report(config.device)
    device_ok = device_report.request_satisfied
    device_check = DoctorCheck(
        name="Device",
        ok=device_ok,
        detail=(
            f"requested={device_report.requested} selected={device_report.selected} "
            f"available={','.join(device_report.available)}"
        ),
    )

    return DoctorReport(
        checks=(python_check, ffmpeg_check, cache_check, device_check),
    )


def render_doctor_report(report: DoctorReport) -> str:
    header = "jamak doctor: OK" if report.ok else "jamak doctor: FAIL"
    lines = [header]
    for check in report.checks:
        status = "PASS" if check.ok else "FAIL"
        lines.append(f"- [{status}] {check.name}: {check.detail}")
    return "\n".join(lines)

