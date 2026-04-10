# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared Granite 3 output functions and base utilities."""

import pytest

from mellea.formatters.granite.base.util import find_substring_in_text
from mellea.formatters.granite.granite3.output import (
    add_citation_context_spans,
    add_hallucination_response_spans,
    create_dict,
    parse_hallucinations_text,
)

# ---------------------------------------------------------------------------
# find_substring_in_text
# ---------------------------------------------------------------------------


class TestFindSubstringInText:
    def test_single_match(self):
        result = find_substring_in_text("world", "hello world")
        assert result == [{"begin_idx": 6, "end_idx": 11}]

    def test_multiple_matches(self):
        result = find_substring_in_text("ab", "ab cd ab ef ab")
        assert len(result) == 3
        assert result[0] == {"begin_idx": 0, "end_idx": 2}
        assert result[1] == {"begin_idx": 6, "end_idx": 8}
        assert result[2] == {"begin_idx": 12, "end_idx": 14}

    def test_no_match(self):
        result = find_substring_in_text("xyz", "hello world")
        assert result == []

    def test_regex_special_chars(self):
        """Characters like ( ) . * should be treated as literals."""
        result = find_substring_in_text("foo(bar)", "call foo(bar) now")
        assert result == [{"begin_idx": 5, "end_idx": 13}]


# ---------------------------------------------------------------------------
# create_dict
# ---------------------------------------------------------------------------


class TestCreateDict:
    def test_single_key(self):
        items = [{"id": "1", "name": "alice"}, {"id": "2", "name": "bob"}]
        result = create_dict(items, key1="id")
        assert result == {
            "1": {"id": "1", "name": "alice"},
            "2": {"id": "2", "name": "bob"},
        }

    def test_compound_key(self):
        items = [
            {"citation_id": "1", "doc_id": "2", "text": "foo"},
            {"citation_id": "1", "doc_id": "3", "text": "bar"},
        ]
        result = create_dict(items, a="citation_id", b="doc_id")
        assert "1-2" in result
        assert "1-3" in result
        assert result["1-2"]["text"] == "foo"

    def test_duplicate_key_warns(self, caplog):
        items = [{"id": "1", "val": "first"}, {"id": "1", "val": "second"}]
        result = create_dict(items, key1="id")
        assert result["1"]["val"] == "second"  # last wins
        assert "duplicate" in caplog.text.lower()

    def test_empty_input(self):
        result = create_dict([], key1="id")
        assert result == {}


# ---------------------------------------------------------------------------
# parse_hallucinations_text
# ---------------------------------------------------------------------------


class TestParseHallucinationsText:
    def test_single_hallucination(self):
        text = "1. Risk high: The sky is green."
        result = parse_hallucinations_text(text)
        assert len(result) == 1
        assert result[0]["hallucination_id"] == "1"
        assert result[0]["risk"] == "high"
        assert "sky is green" in result[0]["response_text"]

    def test_multiple_hallucinations(self):
        text = (
            "1. Risk low: Sentence one.\n"
            "2. Risk high: Sentence two.\n"
            "3. Risk unanswerable: Sentence three."
        )
        result = parse_hallucinations_text(text)
        assert len(result) == 3
        assert result[0]["risk"] == "low"
        assert result[1]["risk"] == "high"
        assert result[2]["risk"] == "unanswerable"

    def test_empty_text_warns(self, caplog):
        result = parse_hallucinations_text("")
        assert result == []
        assert "failed to extract" in caplog.text.lower()

    def test_malformed_text_warns(self, caplog):
        result = parse_hallucinations_text("some random text without format")
        assert result == []


# ---------------------------------------------------------------------------
# add_hallucination_response_spans
# ---------------------------------------------------------------------------


class TestAddHallucinationResponseSpans:
    @staticmethod
    def _identity(text: str) -> str:
        """No-op remove_citations function for testing."""
        return text

    def test_single_match(self):
        hallucination_info = [
            {
                "hallucination_id": "1",
                "risk": "high",
                "response_text": "The sky is green.",
            }
        ]
        response = "Hello world. The sky is green. Goodbye."
        result = add_hallucination_response_spans(
            hallucination_info, response, self._identity
        )
        assert result[0]["response_begin"] == 13
        assert result[0]["response_end"] == 30
        assert result[0]["response_text"] == "The sky is green."

    def test_not_found_uses_placeholder(self, caplog):
        hallucination_info = [
            {
                "hallucination_id": "1",
                "risk": "low",
                "response_text": "This text does not exist.",
            }
        ]
        result = add_hallucination_response_spans(
            hallucination_info, "Completely different response.", self._identity
        )
        assert result[0]["response_begin"] == 0
        assert result[0]["response_end"] == 0

    def test_does_not_mutate_input(self):
        original = [{"hallucination_id": "1", "risk": "high", "response_text": "sky"}]
        add_hallucination_response_spans(original, "the sky is blue", self._identity)
        assert "response_begin" not in original[0]


# ---------------------------------------------------------------------------
# add_citation_context_spans
# ---------------------------------------------------------------------------


class TestAddCitationContextSpans:
    def test_single_citation(self):
        citation_info = [
            {"citation_id": "1", "doc_id": "1", "context_text": "RAG stands for"}
        ]
        docs = [
            {
                "citation_id": "1",
                "doc_id": "1",
                "text": "RAG stands for retrieval-augmented generation.",
            }
        ]
        result = add_citation_context_spans(citation_info, docs)
        assert result[0]["context_begin"] == 0
        assert result[0]["context_end"] == 14

    def test_missing_doc_warns(self, caplog):
        citation_info = [
            {"citation_id": "1", "doc_id": "99", "context_text": "missing"}
        ]
        docs = [{"citation_id": "1", "doc_id": "1", "text": "some text"}]
        result = add_citation_context_spans(citation_info, docs)
        assert result[0]["context_begin"] == 0
        assert result[0]["context_end"] == 0
        assert "not found" in caplog.text.lower()

    def test_text_not_found_in_doc_warns(self, caplog):
        citation_info = [
            {"citation_id": "1", "doc_id": "1", "context_text": "nonexistent phrase"}
        ]
        docs = [{"citation_id": "1", "doc_id": "1", "text": "Completely unrelated."}]
        result = add_citation_context_spans(citation_info, docs)
        assert result[0]["context_begin"] == 0
        assert result[0]["context_end"] == 0

    def test_does_not_mutate_input(self):
        original = [{"citation_id": "1", "doc_id": "1", "context_text": "hello"}]
        docs = [{"citation_id": "1", "doc_id": "1", "text": "hello world"}]
        add_citation_context_spans(original, docs)
        assert "context_begin" not in original[0]
