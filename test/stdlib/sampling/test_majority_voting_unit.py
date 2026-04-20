"""Unit tests for majority voting compare_strings methods — no backend required."""

import pytest

from mellea.stdlib.sampling.majority_voting import (
    MajorityVotingStrategyForMath,
    MBRDRougeLStrategy,
)

# --- MajorityVotingStrategyForMath.compare_strings ---


@pytest.fixture
def math_strategy():
    return MajorityVotingStrategyForMath()


def test_math_compare_identical_boxed(math_strategy):
    assert math_strategy.compare_strings(r"\boxed{2}", r"\boxed{2}") == 1.0


def test_math_compare_identical_latex(math_strategy):
    assert math_strategy.compare_strings(r"\boxed{4}", r"\boxed{4}") == 1.0


def test_math_compare_unboxed_integers_return_zero(math_strategy):
    # Plain integers without boxed notation are not extracted — returns 0.0
    assert math_strategy.compare_strings("2", "3") == 0.0


def test_math_compare_different_boxed(math_strategy):
    assert math_strategy.compare_strings(r"\boxed{2}", r"\boxed{3}") == 0.0


def test_math_compare_returns_float(math_strategy):
    result = math_strategy.compare_strings(r"\boxed{5}", r"\boxed{5}")
    assert isinstance(result, float)


# --- MBRDRougeLStrategy.compare_strings ---


@pytest.fixture
def rouge_strategy():
    return MBRDRougeLStrategy()


def test_rougel_compare_identical(rouge_strategy):
    score = rouge_strategy.compare_strings("hello world", "hello world")
    assert score == pytest.approx(1.0)


def test_rougel_compare_completely_different(rouge_strategy):
    score = rouge_strategy.compare_strings("hello world", "foo bar baz")
    assert score < 0.5


def test_rougel_compare_partial_overlap(rouge_strategy):
    score = rouge_strategy.compare_strings("the quick brown fox", "the quick fox")
    assert 0.0 < score < 1.0


def test_rougel_compare_returns_float(rouge_strategy):
    score = rouge_strategy.compare_strings("abc", "abc")
    assert isinstance(score, float)


def test_rougel_score_in_range(rouge_strategy):
    score = rouge_strategy.compare_strings("some text here", "some different text")
    assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
