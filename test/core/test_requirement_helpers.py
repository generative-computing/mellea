"""Unit tests for core/requirement.py pure helpers — ValidationResult, default_output_to_bool."""

import pytest

from mellea.core import CBlock, ModelOutputThunk
from mellea.core.requirement import ValidationResult, default_output_to_bool

# --- ValidationResult ---


def test_validation_result_pass():
    r = ValidationResult(result=True)
    assert r.as_bool() is True
    assert bool(r) is True


def test_validation_result_fail():
    r = ValidationResult(result=False)
    assert r.as_bool() is False
    assert bool(r) is False


def test_validation_result_reason():
    r = ValidationResult(result=True, reason="looks good")
    assert r.reason == "looks good"


def test_validation_result_score():
    r = ValidationResult(result=True, score=0.95)
    assert r.score == pytest.approx(0.95)


def test_validation_result_thunk():
    mot = ModelOutputThunk(value="x")
    r = ValidationResult(result=True, thunk=mot)
    assert r.thunk is mot


def test_validation_result_context():
    from mellea.stdlib.context import SimpleContext

    ctx = SimpleContext()
    r = ValidationResult(result=True, context=ctx)
    assert r.context is ctx


def test_validation_result_defaults_none():
    r = ValidationResult(result=False)
    assert r.reason is None
    assert r.score is None
    assert r.thunk is None
    assert r.context is None


# --- default_output_to_bool ---


def test_yes_exact_passes():
    assert default_output_to_bool(CBlock("yes")) is True


def test_yes_uppercase_passes():
    assert default_output_to_bool(CBlock("YES")) is True


def test_y_passes():
    assert default_output_to_bool(CBlock("y")) is True


def test_yes_in_sentence():
    assert default_output_to_bool(CBlock("Yes, it meets the requirement.")) is True


def test_no_fails():
    assert default_output_to_bool(CBlock("no")) is False


def test_empty_string_fails():
    assert default_output_to_bool(CBlock("")) is False


def test_random_text_fails():
    assert default_output_to_bool(CBlock("the output looks reasonable")) is False


def test_plain_string_yes():
    assert default_output_to_bool("YES") is True  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
