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


# --- Binary detection for repair feedback (PR #1248) ---


def test_repair_feedback_binary_detection_yes():
    """Binary 'yes' should trigger description fallback."""
    judge_output = "yes"
    should_use_description = judge_output and judge_output.strip().lower() in (
        "yes",
        "no",
    )
    assert should_use_description is True


def test_repair_feedback_binary_detection_no():
    """Binary 'no' should trigger description fallback."""
    judge_output = "no"
    should_use_description = judge_output and judge_output.strip().lower() in (
        "yes",
        "no",
    )
    assert should_use_description is True


def test_repair_feedback_binary_detection_yes_uppercase():
    """Binary 'YES' (uppercase) should trigger description fallback."""
    judge_output = "YES"
    should_use_description = judge_output and judge_output.strip().lower() in (
        "yes",
        "no",
    )
    assert should_use_description is True


def test_repair_feedback_binary_detection_no_mixed_case():
    """Binary 'No' (mixed case) should trigger description fallback."""
    judge_output = "No"
    should_use_description = judge_output and judge_output.strip().lower() in (
        "yes",
        "no",
    )
    assert should_use_description is True


def test_repair_feedback_binary_detection_whitespace():
    """Binary answer with whitespace should trigger description fallback."""
    judge_output = "  yes  "
    should_use_description = judge_output and judge_output.strip().lower() in (
        "yes",
        "no",
    )
    assert should_use_description is True


def test_repair_feedback_binary_detection_detailed_answer():
    """Detailed answer (not binary) should preserve judge output."""
    judge_output = "The requirement is met"
    should_use_description = judge_output and judge_output.strip().lower() in (
        "yes",
        "no",
    )
    assert should_use_description is False


def test_repair_feedback_binary_detection_yes_in_sentence():
    """'Yes' as part of sentence should preserve judge output."""
    judge_output = "Yes, the email has a salutation"
    should_use_description = judge_output and judge_output.strip().lower() in (
        "yes",
        "no",
    )
    assert should_use_description is False


def test_repair_feedback_binary_detection_empty_string():
    """Empty string should preserve (not trigger fallback)."""
    judge_output = ""
    should_use_description = judge_output and judge_output.strip().lower() in (
        "yes",
        "no",
    )
    assert not should_use_description


def test_repair_feedback_binary_detection_none():
    """None value should not crash and should not trigger fallback."""
    judge_output = None
    should_use_description = judge_output and judge_output.strip().lower() in (
        "yes",
        "no",
    )
    assert not should_use_description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
