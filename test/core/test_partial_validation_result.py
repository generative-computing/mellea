"""Unit tests for PartialValidationResult tri-state semantics."""

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


def test_as_bool_matches_bool():
    for state in ("pass", "fail", "unknown"):
        pvr = PartialValidationResult(state)  # type: ignore[arg-type]
        assert pvr.as_bool() == bool(pvr)
