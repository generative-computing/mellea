"""Unit tests for formatters/granite/base/util.py pure helpers."""

import pytest

from mellea.formatters.granite.base.util import find_substring_in_text

# --- find_substring_in_text ---


def test_find_single_match():
    result = find_substring_in_text("hello", "say hello world")
    assert len(result) == 1
    assert result[0]["begin_idx"] == 4
    assert result[0]["end_idx"] == 9


def test_find_multiple_matches():
    result = find_substring_in_text("ab", "ababab")
    assert len(result) == 3
    # Verify positions are non-overlapping
    assert result[0]["begin_idx"] == 0
    assert result[1]["begin_idx"] == 2
    assert result[2]["begin_idx"] == 4


def test_find_no_match_returns_empty():
    result = find_substring_in_text("xyz", "hello world")
    assert result == []


def test_find_empty_text_returns_empty():
    result = find_substring_in_text("hello", "")
    assert result == []


def test_find_at_start():
    result = find_substring_in_text("the", "the quick fox")
    assert result[0]["begin_idx"] == 0


def test_find_at_end():
    result = find_substring_in_text("fox", "the quick fox")
    assert result[-1]["end_idx"] == len("the quick fox")


def test_find_full_text_match():
    result = find_substring_in_text("exact", "exact")
    assert len(result) == 1
    assert result[0]["begin_idx"] == 0
    assert result[0]["end_idx"] == 5


def test_find_special_regex_chars_escaped():
    # Dots in the substring should be treated literally
    result = find_substring_in_text("a.b", "a.b and axb")
    assert len(result) == 1
    assert result[0]["begin_idx"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
