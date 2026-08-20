# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the rag adapter functions' output contracts (Epic #929, #1516).

The contracts are declared in `mellea.backends.adapters.io_contracts` and
looked up by catalog name via `get_io_contract`.  Tests the `parse()` method
of each contract directly — no backend, no GPU, no model download required.
Two tests per helper:

- `test_<helper>_contract_enforced` — output missing a required field raises
  :class:`~mellea.backends.adapters.AdapterSchemaMismatchError`.
- `test_<helper>_forward_compat` — output containing an extra optional field
  does *not* raise.
"""

import json

import pytest

from mellea.backends.adapters import AdapterSchemaMismatchError, get_io_contract

# ---------------------------------------------------------------------------
# check_answerability
# ---------------------------------------------------------------------------


def test_check_answerability_contract_enforced() -> None:
    contract = get_io_contract("answerability")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"wrong_key": "value"}))
    err = exc_info.value
    assert err.name == "answerability"
    assert "answerability" in err.expected_keys


def test_check_answerability_forward_compat() -> None:
    contract = get_io_contract("answerability")
    result = contract.parse(
        json.dumps({"answerability": "answerable", "extra": "ignored"})
    )
    assert result["answerability"] == "answerable"


# ---------------------------------------------------------------------------
# rewrite_question
# ---------------------------------------------------------------------------


def test_rewrite_question_contract_enforced() -> None:
    contract = get_io_contract("query_rewrite")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"wrong_key": "value"}))
    err = exc_info.value
    assert err.name == "query_rewrite"
    assert "rewritten_question" in err.expected_keys


def test_rewrite_question_forward_compat() -> None:
    contract = get_io_contract("query_rewrite")
    result = contract.parse(
        json.dumps({"rewritten_question": "new query?", "confidence": 0.9})
    )
    assert result["rewritten_question"] == "new query?"


# ---------------------------------------------------------------------------
# clarify_query
# ---------------------------------------------------------------------------


def test_clarify_query_contract_enforced() -> None:
    contract = get_io_contract("query_clarification")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"wrong_key": "value"}))
    err = exc_info.value
    assert err.name == "query_clarification"
    assert "clarification" in err.expected_keys


def test_clarify_query_forward_compat() -> None:
    contract = get_io_contract("query_clarification")
    result = contract.parse(json.dumps({"clarification": "CLEAR", "score": 1.0}))
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
    contract = get_io_contract("citations")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps([bad_item]))
    err = exc_info.value
    assert err.name == "citations"
    assert "citation_doc_id" in err.expected_keys


def test_find_citations_forward_compat() -> None:
    extra_item = {**_GOOD_CITATION, "extra_field": "ignored"}
    contract = get_io_contract("citations")
    result = contract.parse(json.dumps([extra_item]))
    assert result["items"][0]["citation_doc_id"] == "0"  # type: ignore[index]


# ---------------------------------------------------------------------------
# check_context_relevance
# ---------------------------------------------------------------------------


def test_check_context_relevance_contract_enforced() -> None:
    contract = get_io_contract("context_relevance")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"wrong_key": "value"}))
    err = exc_info.value
    assert err.name == "context_relevance"
    assert "context_relevance" in err.expected_keys


def test_check_context_relevance_forward_compat() -> None:
    contract = get_io_contract("context_relevance")
    result = contract.parse(json.dumps({"context_relevance": "relevant", "score": 0.8}))
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
    contract = get_io_contract("hallucination_detection")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps([bad_item]))
    err = exc_info.value
    assert err.name == "hallucination_detection"
    assert "explanation" in err.expected_keys


def test_flag_hallucinated_content_forward_compat() -> None:
    extra_item = {**_GOOD_SPAN, "extra_field": "ignored"}
    contract = get_io_contract("hallucination_detection")
    result = contract.parse(json.dumps([extra_item]))
    assert result["items"][0]["faithfulness"] == "faithful"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Empty-list edge cases for list-shaped contracts
# ---------------------------------------------------------------------------


def test_find_citations_empty_list() -> None:
    contract = get_io_contract("citations")
    result = contract.parse(json.dumps([]))
    assert result == {"items": []}


def test_flag_hallucinated_content_empty_list() -> None:
    contract = get_io_contract("hallucination_detection")
    result = contract.parse(json.dumps([]))
    assert result == {"items": []}


# ---------------------------------------------------------------------------
# Type-mismatch: ValueError raised when JSON is the wrong shape
# ---------------------------------------------------------------------------


def test_dict_contract_rejects_non_dict() -> None:
    contract = get_io_contract("answerability")
    with pytest.raises(ValueError, match="must be a JSON object"):
        contract.parse(json.dumps(["not", "a", "dict"]))


def test_dict_contract_error_mentions_adapter_name() -> None:
    contract = get_io_contract("answerability")
    with pytest.raises(ValueError, match="answerability"):
        contract.parse(json.dumps(42))


def test_list_contract_rejects_non_list() -> None:
    contract = get_io_contract("citations")
    with pytest.raises(ValueError, match="must be a JSON array"):
        contract.parse(json.dumps({"not": "a list"}))


def test_list_contract_rejects_non_dict_element() -> None:
    contract = get_io_contract("citations")
    with pytest.raises(ValueError, match="must contain only JSON objects"):
        contract.parse(json.dumps(["string_element"]))


def test_list_contract_rejects_non_dict_element_after_valid_item() -> None:
    contract = get_io_contract("citations")
    with pytest.raises(ValueError, match="must contain only JSON objects"):
        contract.parse(json.dumps([_GOOD_CITATION, "string_element"]))
