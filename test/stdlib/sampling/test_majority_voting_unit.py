# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for majority voting compare_strings methods — no backend required."""

import pytest

from mellea.stdlib.sampling import majority_voting
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


def test_math_compare_different_unboxed_integers_return_zero(math_strategy):
    assert math_strategy.compare_strings("2", "3") == 0.0


def test_math_compare_equal_unboxed_integers(math_strategy):
    # Bare expressions are extracted via the "expr" match type, so two samples
    # that both answered a plain `2` agree.
    assert math_strategy.compare_strings("2", "2") == 1.0


def test_math_compare_different_boxed(math_strategy):
    assert math_strategy.compare_strings(r"\boxed{2}", r"\boxed{3}") == 0.0


def test_math_compare_returns_float(math_strategy):
    result = math_strategy.compare_strings(r"\boxed{5}", r"\boxed{5}")
    assert isinstance(result, float)


# --- MajorityVotingStrategyForMath extraction-target cache ---


def test_math_match_types_default(math_strategy):
    assert math_strategy.match_types == ["latex", "expr"]


def test_math_compare_respects_mutated_match_types(math_strategy):
    # The cached extraction targets must follow the public `match_types`.
    math_strategy.match_types[:] = ["latex"]
    assert math_strategy.compare_strings("1+1", "2") == 0.0


def test_math_compare_respects_reassigned_match_types(math_strategy):
    math_strategy.match_types = ["latex"]
    assert math_strategy.compare_strings("1+1", "2") == 0.0


def test_math_compare_reverts_when_match_types_restored(math_strategy):
    math_strategy.match_types[:] = ["latex"]
    math_strategy.compare_strings("1+1", "2")
    math_strategy.match_types[:] = ["latex", "expr"]
    assert math_strategy.compare_strings("1+1", "2") == 1.0


def test_math_compare_does_not_rebuild_targets_when_unchanged(
    math_strategy, monkeypatch
):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        majority_voting,
        "_build_extraction_targets",
        lambda match_types: calls.append(match_types) or [],
    )
    math_strategy.compare_strings(r"\boxed{2}", r"\boxed{2}")
    math_strategy.compare_strings(r"\boxed{3}", r"\boxed{3}")

    assert calls == []


def test_math_compare_rebuilds_targets_when_match_types_change(
    math_strategy, monkeypatch
):
    calls: list[tuple[str, ...]] = []
    real = majority_voting._build_extraction_targets

    def counted(match_types):
        calls.append(match_types)
        return real(match_types)

    monkeypatch.setattr(majority_voting, "_build_extraction_targets", counted)
    math_strategy.match_types[:] = ["latex"]
    math_strategy.compare_strings("1+1", "2")

    assert calls == [("latex",)]


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
