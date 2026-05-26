"""Unit tests for SOFAI sampling strategy pure static helpers — no backend required.

Covers _extract_action_prompt, _parse_judgment, _extract_feedback, _select_best_attempt.
"""

import pytest

from mellea.core import Requirement, TemplateRepresentation, ValidationResult
from mellea.stdlib.components import Instruction, Message
from mellea.stdlib.sampling.sofai import SOFAISamplingStrategy

# --- _parse_judgment ---


def test_parse_judgment_yes():
    assert SOFAISamplingStrategy._parse_judgment("Yes") is True


def test_parse_judgment_yes_with_explanation():
    assert SOFAISamplingStrategy._parse_judgment("Yes, the output is correct.") is True


def test_parse_judgment_no():
    assert SOFAISamplingStrategy._parse_judgment("No") is False


def test_parse_judgment_no_with_explanation():
    assert (
        SOFAISamplingStrategy._parse_judgment(
            "No, it needs improvement.\nDetails here."
        )
        is False
    )


def test_parse_judgment_yes_in_first_line():
    assert SOFAISamplingStrategy._parse_judgment("The answer is yes") is True


def test_parse_judgment_no_match_defaults_false():
    assert SOFAISamplingStrategy._parse_judgment("Maybe, hard to tell") is False


def test_parse_judgment_whitespace_stripped():
    assert SOFAISamplingStrategy._parse_judgment("  Yes  ") is True


def test_parse_judgment_case_insensitive():
    assert SOFAISamplingStrategy._parse_judgment("YES") is True


# --- _extract_feedback ---


def test_extract_feedback_with_tags():
    text = "Some preamble. <feedback>Fix the grammar.</feedback> More text."
    assert SOFAISamplingStrategy._extract_feedback(text) == "Fix the grammar."


def test_extract_feedback_no_tags():
    text = "Just plain feedback text."
    assert SOFAISamplingStrategy._extract_feedback(text) == "Just plain feedback text."


def test_extract_feedback_multiline():
    text = "<feedback>\nLine 1\nLine 2\n</feedback>"
    result = SOFAISamplingStrategy._extract_feedback(text)
    assert "Line 1" in result
    assert "Line 2" in result


def test_extract_feedback_case_insensitive_tags():
    text = "<FEEDBACK>Fix it.</FEEDBACK>"
    assert SOFAISamplingStrategy._extract_feedback(text) == "Fix it."


def test_extract_feedback_strips_whitespace():
    text = "  some feedback  "
    assert SOFAISamplingStrategy._extract_feedback(text) == "some feedback"


# --- _extract_action_prompt ---


def test_extract_action_prompt_message():
    msg = Message("user", "What is 2+2?")
    assert SOFAISamplingStrategy._extract_action_prompt(msg) == "What is 2+2?"


def test_extract_action_prompt_instruction():
    ins = Instruction(description="Summarise the text")
    result = SOFAISamplingStrategy._extract_action_prompt(ins)
    assert result == "Summarise the text"


def test_extract_action_prompt_format_for_llm_str():
    """Component whose format_for_llm returns a plain string."""
    from mellea.core import CBlock, Component, ModelOutputThunk

    class _StrComponent(Component[str]):
        def parts(self):
            return []

        def format_for_llm(self) -> str:
            return "plain text repr"

        def _parse(self, computed: ModelOutputThunk) -> str:
            return ""

    result = SOFAISamplingStrategy._extract_action_prompt(_StrComponent())
    assert result == "plain text repr"


# --- _select_best_attempt ---


def _vr(passed: bool) -> ValidationResult:
    return ValidationResult(result=passed)


def test_select_best_attempt_picks_most_passing():
    r = Requirement(description="r")
    val = [
        [(r, _vr(True)), (r, _vr(False))],  # 1 pass
        [(r, _vr(True)), (r, _vr(True))],  # 2 pass — best
        [(r, _vr(False)), (r, _vr(False))],  # 0 pass
    ]
    assert SOFAISamplingStrategy._select_best_attempt(val) == 1


def test_select_best_attempt_tie_prefers_later():
    r = Requirement(description="r")
    val = [
        [(r, _vr(True))],  # 1 pass
        [(r, _vr(True))],  # 1 pass — tie, but later → preferred
    ]
    assert SOFAISamplingStrategy._select_best_attempt(val) == 1


def test_select_best_attempt_single():
    r = Requirement(description="r")
    val = [[(r, _vr(False))]]
    assert SOFAISamplingStrategy._select_best_attempt(val) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
