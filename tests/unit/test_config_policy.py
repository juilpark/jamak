from jamak.infra.config import evaluate_python_policy


def test_python_policy_accepts_preferred_version() -> None:
    evaluation = evaluate_python_policy((3, 14))
    assert evaluation.status == "preferred"


def test_python_policy_rejects_below_minimum() -> None:
    evaluation = evaluate_python_policy((3, 11))
    assert evaluation.status == "unsupported"

