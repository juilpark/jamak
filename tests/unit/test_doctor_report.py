from jamak.infra.config import build_app_config
from jamak.infra.doctor import collect_doctor_report


def test_doctor_report_has_required_checks() -> None:
    report = collect_doctor_report(build_app_config())
    names = {check.name for check in report.checks}
    assert names == {"Python policy", "ffmpeg", "HuggingFace cache", "Device"}

