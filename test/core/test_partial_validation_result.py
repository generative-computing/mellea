"""Unit tests for PartialValidationResult tri-state semantics."""

import pytest

from mellea.core import PartialValidationResult


def test_pass_state():
    pvr = PartialValidationResult("pass")
    assert pvr.success == "pass"
    assert pvr.as_bool() is True
    assert bool(pvr) is True


def test_fail_state():
    pvr = PartialValidationResult("fail")
    assert pvr.success == "fail"
    assert pvr.as_bool() is False
    assert bool(pvr) is False


def test_unknown_state():
    pvr = PartialValidationResult("unknown")
    assert pvr.success == "unknown"
    assert pvr.as_bool() is False
    assert bool(pvr) is False


def test_default_optional_fields_are_none():
    pvr = PartialValidationResult("unknown")
    assert pvr.reason is None
    assert pvr.score is None
    assert pvr.thunk is None
    assert pvr.context is None


def test_reason_field():
    pvr = PartialValidationResult("fail", reason="Too short")
    assert pvr.reason == "Too short"


def test_score_field():
    pvr = PartialValidationResult("pass", score=0.95)
    assert pvr.score == 0.95


@pytest.mark.parametrize(
    ("state", "expected"), [("pass", True), ("fail", False), ("unknown", False)]
)
def test_as_bool_correctness(state: str, expected: bool) -> None:
    pvr = PartialValidationResult(state)  # type: ignore[arg-type]
    assert pvr.as_bool() is expected
    assert bool(pvr) is expected


def test_invalid_success_raises() -> None:
    with pytest.raises(ValueError, match="success must be"):
        PartialValidationResult("maybe")  # type: ignore[arg-type]


def test_repr_shows_state() -> None:
    pvr = PartialValidationResult("fail", reason="too short", score=0.1)
    r = repr(pvr)
    assert "'fail'" in r
    assert "too short" in r
    assert "0.1" in r
