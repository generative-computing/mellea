"""Unit tests for IOContract validation in rag.py (Epic #929 Phase 1).

Tests the ``parse()`` method of each IOContract subclass directly — no backend,
no GPU, no model download required.  Two tests per helper:

- ``test_<helper>_contract_enforced`` — output missing a required field raises
  :class:`~mellea.backends.adapters.AdapterSchemaMismatchError`.
- ``test_<helper>_forward_compat`` — output containing an extra optional field
  does *not* raise.
"""

import json

import pytest

from mellea.backends.adapters import AdapterSchemaMismatchError
from mellea.stdlib.components.intrinsic.rag import (
    _ANSWERABILITY_ADAPTER,
    _CITATIONS_ADAPTER,
    _CONTEXT_RELEVANCE_ADAPTER,
    _HALLUCINATION_ADAPTER,
    _QUERY_CLARIFY_ADAPTER,
    _QUERY_REWRITE_ADAPTER,
)

# ---------------------------------------------------------------------------
# check_answerability
# ---------------------------------------------------------------------------


def test_check_answerability_contract_enforced() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _ANSWERABILITY_ADAPTER.io_contract.parse(json.dumps({"wrong_key": "value"}))
    err = exc_info.value
    assert err.name == "answerability"
    assert "answerability" in err.expected_keys


def test_check_answerability_forward_compat() -> None:
    result = _ANSWERABILITY_ADAPTER.io_contract.parse(
        json.dumps({"answerability": "answerable", "extra": "ignored"})
    )
    assert result["answerability"] == "answerable"


# ---------------------------------------------------------------------------
# rewrite_question
# ---------------------------------------------------------------------------


def test_rewrite_question_contract_enforced() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _QUERY_REWRITE_ADAPTER.io_contract.parse(json.dumps({"wrong_key": "value"}))
    err = exc_info.value
    assert err.name == "query_rewrite"
    assert "rewritten_question" in err.expected_keys


def test_rewrite_question_forward_compat() -> None:
    result = _QUERY_REWRITE_ADAPTER.io_contract.parse(
        json.dumps({"rewritten_question": "new query?", "confidence": 0.9})
    )
    assert result["rewritten_question"] == "new query?"


# ---------------------------------------------------------------------------
# clarify_query
# ---------------------------------------------------------------------------


def test_clarify_query_contract_enforced() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _QUERY_CLARIFY_ADAPTER.io_contract.parse(json.dumps({"wrong_key": "value"}))
    err = exc_info.value
    assert err.name == "query_clarification"
    assert "clarification" in err.expected_keys


def test_clarify_query_forward_compat() -> None:
    result = _QUERY_CLARIFY_ADAPTER.io_contract.parse(
        json.dumps({"clarification": "CLEAR", "score": 1.0})
    )
    assert result["clarification"] == "CLEAR"


# ---------------------------------------------------------------------------
# find_citations
# ---------------------------------------------------------------------------


_GOOD_CITATION = {
    "response_begin": 0,
    "response_end": 10,
    "response_text": "some text",
    "citation_doc_id": "0",
    "citation_begin": 5,
    "citation_end": 20,
    "citation_text": "source",
}


def test_find_citations_contract_enforced() -> None:
    bad_item = {k: v for k, v in _GOOD_CITATION.items() if k != "citation_doc_id"}
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _CITATIONS_ADAPTER.io_contract.parse(json.dumps([bad_item]))
    err = exc_info.value
    assert err.name == "citations"
    assert "citation_doc_id" in err.expected_keys


def test_find_citations_forward_compat() -> None:
    extra_item = {**_GOOD_CITATION, "extra_field": "ignored"}
    result = _CITATIONS_ADAPTER.io_contract.parse(json.dumps([extra_item]))
    assert result["items"][0]["citation_doc_id"] == "0"  # type: ignore[index]


# ---------------------------------------------------------------------------
# check_context_relevance
# ---------------------------------------------------------------------------


def test_check_context_relevance_contract_enforced() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _CONTEXT_RELEVANCE_ADAPTER.io_contract.parse(json.dumps({"wrong_key": "value"}))
    err = exc_info.value
    assert err.name == "context_relevance"
    assert "context_relevance" in err.expected_keys


def test_check_context_relevance_forward_compat() -> None:
    result = _CONTEXT_RELEVANCE_ADAPTER.io_contract.parse(
        json.dumps({"context_relevance": "relevant", "score": 0.8})
    )
    assert result["context_relevance"] == "relevant"


# ---------------------------------------------------------------------------
# flag_hallucinated_content
# ---------------------------------------------------------------------------


_GOOD_SPAN = {
    "response_begin": 0,
    "response_end": 10,
    "response_text": "some text",
    "faithfulness": "faithful",
    "explanation": "supported by document",
}


def test_flag_hallucinated_content_contract_enforced() -> None:
    bad_item = {k: v for k, v in _GOOD_SPAN.items() if k != "explanation"}
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _HALLUCINATION_ADAPTER.io_contract.parse(json.dumps([bad_item]))
    err = exc_info.value
    assert err.name == "hallucination_detection"
    assert "explanation" in err.expected_keys


def test_flag_hallucinated_content_forward_compat() -> None:
    extra_item = {**_GOOD_SPAN, "extra_field": "ignored"}
    result = _HALLUCINATION_ADAPTER.io_contract.parse(json.dumps([extra_item]))
    assert result["items"][0]["faithfulness"] == "faithful"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Empty-list edge cases for list-shaped contracts
# ---------------------------------------------------------------------------


def test_find_citations_empty_list() -> None:
    result = _CITATIONS_ADAPTER.io_contract.parse(json.dumps([]))
    assert result == {"items": []}


def test_flag_hallucinated_content_empty_list() -> None:
    result = _HALLUCINATION_ADAPTER.io_contract.parse(json.dumps([]))
    assert result == {"items": []}


# ---------------------------------------------------------------------------
# Type-mismatch: ValueError raised when JSON is the wrong shape
# ---------------------------------------------------------------------------


def test_dict_contract_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        _ANSWERABILITY_ADAPTER.io_contract.parse(json.dumps(["not", "a", "dict"]))


def test_dict_contract_error_mentions_adapter_name() -> None:
    with pytest.raises(ValueError, match="answerability"):
        _ANSWERABILITY_ADAPTER.io_contract.parse(json.dumps(42))


def test_list_contract_rejects_non_list() -> None:
    with pytest.raises(ValueError, match="must be a JSON array"):
        _CITATIONS_ADAPTER.io_contract.parse(json.dumps({"not": "a list"}))


def test_list_contract_rejects_non_dict_element() -> None:
    with pytest.raises(ValueError, match="must contain only JSON objects"):
        _CITATIONS_ADAPTER.io_contract.parse(json.dumps(["string_element"]))
