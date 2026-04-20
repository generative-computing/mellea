"""Unit tests for eval runner pure-logic helpers — no backend, no model required.

Covers InputEvalResult, TestEvalResult, parse_judge_output.
"""

import pytest

from cli.eval.runner import InputEvalResult, TestEvalResult, parse_judge_output
from mellea.stdlib.components.unit_test_eval import TestBasedEval

# --- InputEvalResult ---


def test_input_eval_result_to_dict():
    r = InputEvalResult(
        input_text="What is 2+2?",
        model_output="4",
        validation_passed=True,
        score=1,
        validation_reason="Correct answer",
    )
    d = r.to_dict()
    assert d["input"] == "What is 2+2?"
    assert d["model_output"] == "4"
    assert d["passed"] is True
    assert d["score"] == 1
    assert d["justification"] == "Correct answer"


def test_input_eval_result_to_dict_failed():
    r = InputEvalResult("q", "wrong", False, 0, "Incorrect")
    d = r.to_dict()
    assert d["passed"] is False
    assert d["score"] == 0


# --- TestEvalResult ---


def _make_test_eval() -> TestBasedEval:
    return TestBasedEval(
        source="test_source",
        name="test_name",
        instructions="Judge if correct",
        inputs=["input1", "input2"],
        test_id="test-001",
    )


def _make_input_results(passed: list[bool]) -> list[InputEvalResult]:
    return [
        InputEvalResult(f"q{i}", f"a{i}", p, 1 if p else 0, "reason")
        for i, p in enumerate(passed)
    ]


def test_test_eval_result_passed_count():
    eval_spec = _make_test_eval()
    results = _make_input_results([True, False])
    r = TestEvalResult(eval_spec, results)
    assert r.passed_count == 1


def test_test_eval_result_pass_rate():
    eval_spec = _make_test_eval()
    results = _make_input_results([True, False])
    r = TestEvalResult(eval_spec, results)
    assert r.pass_rate == pytest.approx(0.5)


def test_test_eval_result_pass_rate_empty():
    eval_spec = _make_test_eval()
    r = TestEvalResult(eval_spec, [])
    assert r.pass_rate == 0.0


def test_test_eval_result_all_pass():
    eval_spec = _make_test_eval()
    results = _make_input_results([True, True])
    r = TestEvalResult(eval_spec, results)
    assert r.pass_rate == pytest.approx(1.0)


def test_test_eval_result_to_dict_structure():
    eval_spec = _make_test_eval()
    results = _make_input_results([True, False])
    r = TestEvalResult(eval_spec, results)
    d = r.to_dict()
    assert d["test_id"] == "test-001"
    assert d["source"] == "test_source"
    assert d["name"] == "test_name"
    assert d["instructions"] == "Judge if correct"
    assert len(d["input_results"]) == 2
    assert d["passed"] == 1
    assert d["total_count"] == 2
    assert d["pass_rate"] == pytest.approx(0.5)


# --- parse_judge_output ---


def test_parse_json_score_and_justification():
    output = '{"score": 1, "justification": "Correct answer"}'
    score, reason = parse_judge_output(output)
    assert score == 1
    assert reason == "Correct answer"


def test_parse_json_embedded_in_text():
    output = 'Based on my review: {"score": 0, "justification": "Wrong answer"} end.'
    score, reason = parse_judge_output(output)
    assert score == 0
    assert reason == "Wrong answer"


def test_parse_score_from_plain_text():
    output = "Score: 1\nThe answer is correct."
    score, reason = parse_judge_output(output)
    assert score == 1
    assert reason == output


def test_parse_no_score_returns_none():
    output = "I cannot determine the score."
    score, reason = parse_judge_output(output)
    assert score is None
    assert reason == output


def test_parse_invalid_json_falls_back_to_regex():
    output = 'Almost JSON: {"score": 1, but broken}'
    score, reason = parse_judge_output(output)
    # Regex fallback should find "score": 1 and return the full raw text as justification
    assert score == 1
    assert reason == output


def test_parse_zero_score():
    output = '{"score": 0, "justification": "Failed"}'
    score, reason = parse_judge_output(output)
    assert score == 0
    assert reason == "Failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
