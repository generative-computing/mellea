# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the canonical adapter-function output-contract registry (issue #1516).

Covers the registry's completeness/exhaustiveness invariant, the fallback contract
for names outside the catalog, and the `_RequirementCheckContract` parsing logic that
replaced `core.requirement_check`'s hand-rolled post-call validation.
"""

import json
import math
import pathlib

import pytest

from mellea.backends.adapters import AdapterSchemaMismatchError, catalog
from mellea.backends.adapters.catalog import (
    IntrinsicsCatalogEntry,
    known_intrinsic_names,
)
from mellea.backends.adapters.io_contracts import (
    _BUILTIN_INTRINSIC_NAMES,
    _INTRINSIC_IO_CONTRACTS,
    get_io_contract,
)
from mellea.formatters.granite import (
    ChatCompletion,
    ChatCompletionResponse,
    IntrinsicsResultProcessor,
)
from mellea.formatters.granite.intrinsics.output import DecodeSentences

_INTRINSIC_TESTDATA = (
    pathlib.Path(__file__).resolve().parents[2]
    / "stdlib"
    / "components"
    / "intrinsic"
    / "testdata"
)
_GRANITE_TESTDATA = (
    pathlib.Path(__file__).resolve().parents[2] / "formatters" / "granite" / "testdata"
)

# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_registry_covers_every_catalog_name():
    """Every catalogued adapter function must have a declared contract.

    Regression guard for the acceptance criterion that no adapter reachable
    through `resolve_adapter` carries an unimplemented placeholder contract.
    """
    missing = _BUILTIN_INTRINSIC_NAMES - set(_INTRINSIC_IO_CONTRACTS)
    assert missing == set()


def test_registry_has_no_orphan_keys():
    """A contract keyed to a name absent from the catalog would be dead code.

    Guards the reverse direction of the completeness invariant: an orphan key
    would be served by `get_io_contract` ahead of the permissive fallback if a
    later adapter ever registered under that name.
    """
    orphans = set(_INTRINSIC_IO_CONTRACTS) - _BUILTIN_INTRINSIC_NAMES
    assert orphans == set()


def test_get_io_contract_returns_declared_instance_for_known_names():
    for name in _BUILTIN_INTRINSIC_NAMES:
        assert get_io_contract(name) is _INTRINSIC_IO_CONTRACTS[name]


def test_registry_invariant_ignores_runtime_custom_catalogue_entry(monkeypatch):
    """A custom adapter must not make the built-in registry checks order-dependent."""
    custom_name = "custom-user-adapter"
    custom_entry = IntrinsicsCatalogEntry(
        name=custom_name, repo_id="example/custom-user-adapter", revision="main"
    )
    monkeypatch.setattr(
        catalog,
        "_INTRINSICS_CATALOG",
        {**catalog._INTRINSICS_CATALOG, custom_name: custom_entry},
    )

    assert custom_name in known_intrinsic_names()
    assert custom_name not in _BUILTIN_INTRINSIC_NAMES
    assert _BUILTIN_INTRINSIC_NAMES - set(_INTRINSIC_IO_CONTRACTS) == set()


def test_get_io_contract_falls_back_permissively_for_unknown_names():
    """A name outside the catalog (a custom, non-catalog adapter) must not raise."""
    contract = get_io_contract("some-custom-user-adapter")
    result = contract.parse(json.dumps({"anything": "goes"}))
    assert result == {"anything": "goes"}


def test_get_io_contract_fallback_still_rejects_non_dict_output():
    contract = get_io_contract("some-custom-user-adapter")
    with pytest.raises(ValueError, match="must be a JSON object"):
        contract.parse(json.dumps(["not", "a", "dict"]))


# ---------------------------------------------------------------------------
# _RequirementCheckContract — replaces core.requirement_check's hand-rolled validation
# ---------------------------------------------------------------------------


@pytest.fixture
def contract():
    return _INTRINSIC_IO_CONTRACTS["requirement-check"]


def test_valid_score_returned(contract):
    result = contract.parse(json.dumps({"requirement_check": {"score": 0.8}}))
    assert result["requirement_check"]["score"] == pytest.approx(0.8)  # type: ignore[index]


def test_missing_requirement_check_key_raises(contract):
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"other_field": 0.9}))


def test_requirement_check_not_a_dict_raises(contract):
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"requirement_check": None}))


def test_requirement_check_list_raises(contract):
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"requirement_check": []}))


def test_missing_score_key_raises(contract):
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"requirement_check": {"other_key": 0.9}}))


def test_null_score_raises(contract):
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"requirement_check": {"score": None}}))


def test_string_score_raises(contract):
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"requirement_check": {"score": "0.9"}}))


def test_bool_score_raises(contract):
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"requirement_check": {"score": True}}))


def test_nan_score_raises(contract):
    # json.dumps(nan) emits the non-standard `NaN` token, which json.loads accepts.
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"requirement_check": {"score": math.nan}}))


def test_inf_score_raises(contract):
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"requirement_check": {"score": math.inf}}))


def test_score_above_range_raises(contract):
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"requirement_check": {"score": 1.5}}))


def test_score_below_range_raises(contract):
    with pytest.raises(AdapterSchemaMismatchError):
        contract.parse(json.dumps({"requirement_check": {"score": -0.1}}))


def test_boundary_score_zero(contract):
    result = contract.parse(json.dumps({"requirement_check": {"score": 0.0}}))
    assert result["requirement_check"]["score"] == pytest.approx(0.0)  # type: ignore[index]


def test_boundary_score_one(contract):
    result = contract.parse(json.dumps({"requirement_check": {"score": 1.0}}))
    assert result["requirement_check"]["score"] == pytest.approx(1.0)  # type: ignore[index]


def test_requirement_check_rejects_non_dict_output(contract):
    with pytest.raises(ValueError, match="must be a JSON object"):
        contract.parse(json.dumps(["not", "a", "dict"]))


# ---------------------------------------------------------------------------
# context-attribution — recorded model output against the real _ListContract
#
# The GPU-gated equivalents (test_find_context_attributions and
# test_find_context_attributions_resolve in test_core.py) are xfail(strict=False)
# for unrelated non-determinism, so they give no CI signal on schema drift. This
# feeds the exact recorded output through the contract without a GPU.
# ---------------------------------------------------------------------------


def test_context_attribution_contract_accepts_recorded_model_output():
    fixture = _INTRINSIC_TESTDATA / "output_json" / "context-attribution.json"
    completion = json.loads(fixture.read_text(encoding="utf-8"))
    raw = completion["choices"][0]["message"]["content"]

    result = _INTRINSIC_IO_CONTRACTS["context-attribution"].parse(raw)

    assert len(result["items"]) == 7  # type: ignore[arg-type]
    assert result["items"][0]["attribution_msg_index"] is None  # type: ignore[index]


def test_context_attribution_contract_rejects_missing_item_key():
    """A record missing a required item key raises AdapterSchemaMismatchError."""
    fixture = _INTRINSIC_TESTDATA / "output_json" / "context-attribution.json"
    completion = json.loads(fixture.read_text(encoding="utf-8"))
    raw = completion["choices"][0]["message"]["content"]
    items = json.loads(raw)
    del items[0]["attribution_text"]

    with pytest.raises(AdapterSchemaMismatchError):
        _INTRINSIC_IO_CONTRACTS["context-attribution"].parse(json.dumps(items))


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ({"other": "value"}, "neither `label` nor `score` was present"),
        ({"label": "Yes", "score": "No"}, "both `label` and `score` were present"),
    ],
)
def test_policy_guardrails_contract_distinguishes_exclusivity_failures(output, reason):
    """Neither-key and both-key output errors must identify the failed condition."""
    contract = _INTRINSIC_IO_CONTRACTS["policy-guardrails"]

    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps(output))

    assert exc_info.value.reason == reason
    assert reason in str(exc_info.value)


# ---------------------------------------------------------------------------
# _HallucinationDetectionContract (issue #1719)
# ---------------------------------------------------------------------------

_BASE_HALLUCINATION_RECORD: dict[str, object] = {
    "response_begin": 0,
    "response_end": 31,
    "response_text": "Purple bumble fish are yellow. ",
    "faithfulness": "faithful",
    "explanation": "Supported by the document.",
}


@pytest.fixture
def hallucination_contract():
    return _INTRINSIC_IO_CONTRACTS["hallucination_detection"]


def test_hallucination_contract_build_prompt_raises(hallucination_contract):
    with pytest.raises(NotImplementedError):
        hallucination_contract.build_prompt()


@pytest.mark.parametrize(
    ("raw_payload", "match"),
    [
        ({"not": "a list"}, "must be a JSON array, got dict"),
        ("just a string", "must be a JSON array, got str"),
        (123, "must be a JSON array, got int"),
        (["not_a_dict"], "must contain only JSON objects, got a str element"),
        ([123], "must contain only JSON objects, got a int element"),
        ([None], "must contain only JSON objects, got a NoneType element"),
    ],
)
def test_hallucination_contract_rejects_non_array_or_non_object_elements(
    hallucination_contract, raw_payload, match
):
    with pytest.raises(ValueError, match=match):
        hallucination_contract.parse(json.dumps(raw_payload))


@pytest.mark.parametrize(
    "missing_key",
    ["response_begin", "response_end", "response_text", "faithfulness", "explanation"],
)
def test_hallucination_contract_missing_required_key_raises(
    hallucination_contract, missing_key
):
    bad = {k: v for k, v in _BASE_HALLUCINATION_RECORD.items() if k != missing_key}
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        hallucination_contract.parse(json.dumps([bad]))
    assert missing_key in exc_info.value.expected_keys
    assert missing_key not in exc_info.value.observed_keys


def test_hallucination_contract_unknown_faithfulness_label_raises(
    hallucination_contract,
):
    bad = {**_BASE_HALLUCINATION_RECORD, "faithfulness": "hallucinated"}
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        hallucination_contract.parse(json.dumps([bad]))
    msg = str(exc_info.value)
    assert "'hallucinated'" in msg
    for valid_label in ("'faithful'", "'partial'", "'unfaithful'", "'NA'"):
        assert valid_label in msg
    assert "upgrade `mellea`" in msg


@pytest.mark.parametrize(
    ("bad_faithfulness", "expected_type"),
    [(None, "NoneType"), (123, "int"), (True, "bool")],
)
def test_hallucination_contract_non_str_faithfulness_raises_without_upgrade_hint(
    hallucination_contract, bad_faithfulness, expected_type
):
    bad = {**_BASE_HALLUCINATION_RECORD, "faithfulness": bad_faithfulness}
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        hallucination_contract.parse(json.dumps([bad]))
    msg = str(exc_info.value)
    assert "`faithfulness` must be a `str`" in msg
    assert f"got {expected_type}" in msg
    assert "upgrade `mellea`" not in msg


@pytest.mark.parametrize(
    ("field", "bad_offset", "expected_type"),
    [
        ("response_begin", "0", "str"),
        ("response_end", "31", "str"),
        ("response_begin", 0.5, "float"),
        ("response_begin", None, "NoneType"),
        ("response_end", None, "NoneType"),
    ],
)
def test_hallucination_contract_non_int_offset_raises(
    hallucination_contract, field, bad_offset, expected_type
):
    bad = {**_BASE_HALLUCINATION_RECORD, field: bad_offset}
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        hallucination_contract.parse(json.dumps([bad]))
    msg = str(exc_info.value)
    assert f"`{field}` must be `int` (not `bool`), got {expected_type}" in msg


@pytest.mark.parametrize(
    ("field", "bool_offset"),
    [
        ("response_begin", False),
        ("response_begin", True),
        ("response_end", False),
        ("response_end", True),
    ],
)
def test_hallucination_contract_bool_offset_raises(
    hallucination_contract, field, bool_offset
):
    bad = {**_BASE_HALLUCINATION_RECORD, field: bool_offset}
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        hallucination_contract.parse(json.dumps([bad]))
    msg = str(exc_info.value)
    assert f"`{field}` must be `int` (not `bool`), got bool" in msg


def test_hallucination_contract_begin_greater_than_end_raises(hallucination_contract):
    bad = {**_BASE_HALLUCINATION_RECORD, "response_begin": 20, "response_end": 10}
    with pytest.raises(AdapterSchemaMismatchError, match=r"response_begin.*<="):
        hallucination_contract.parse(json.dumps([bad]))


@pytest.mark.parametrize("bad_explanation", [None, 123, ["list"]])
def test_hallucination_contract_non_str_explanation_raises(
    hallucination_contract, bad_explanation
):
    bad = {**_BASE_HALLUCINATION_RECORD, "explanation": bad_explanation}
    with pytest.raises(
        AdapterSchemaMismatchError, match="`explanation` must be a `str`"
    ):
        hallucination_contract.parse(json.dumps([bad]))


def test_hallucination_contract_conflicting_duplicate_span_raises(
    hallucination_contract,
):
    first = {**_BASE_HALLUCINATION_RECORD, "faithfulness": "faithful"}
    second = {**_BASE_HALLUCINATION_RECORD, "faithfulness": "unfaithful"}
    with pytest.raises(AdapterSchemaMismatchError, match="conflicting `faithfulness`"):
        hallucination_contract.parse(json.dumps([first, second]))


def test_hallucination_contract_overlapping_records_raise(hallucination_contract):
    rec_a = {
        **_BASE_HALLUCINATION_RECORD,
        "response_begin": 0,
        "response_end": 20,
        "faithfulness": "faithful",
    }
    rec_b = {
        **_BASE_HALLUCINATION_RECORD,
        "response_begin": 10,
        "response_end": 30,
        "faithfulness": "unfaithful",
    }
    with pytest.raises(AdapterSchemaMismatchError, match="overlapping response spans"):
        hallucination_contract.parse(json.dumps([rec_a, rec_b]))


def test_hallucination_contract_passes_adjacent_records_extra_fields_and_empty_list(
    hallucination_contract,
):
    assert hallucination_contract.parse(json.dumps([])) == {"items": []}

    records = [
        {
            "response_begin": 0,
            "response_end": 10,
            "response_text": "Sentence 0.",
            "faithfulness": "faithful",
            "explanation": "Supported.",
            "extra_optional": 1,
        },
        {
            "response_begin": 10,
            "response_end": 25,
            "response_text": "Sentence 1.",
            "faithfulness": "partial",
            "explanation": "Partially supported.",
        },
        {
            "response_begin": 25,
            "response_end": 40,
            "response_text": "Sentence 2.",
            "faithfulness": "unfaithful",
            "explanation": "Not supported.",
        },
        {
            "response_begin": 40,
            "response_end": 50,
            "response_text": "Hello!",
            "faithfulness": "NA",
            "explanation": "Greeting.",
        },
    ]
    parsed = hallucination_contract.parse(json.dumps(records))
    assert parsed == {"items": records}


def test_hallucination_contract_deduplicates_matching_span_and_faithfulness(
    hallucination_contract,
):
    first = {**_BASE_HALLUCINATION_RECORD, "explanation": "First explanation kept."}
    duplicate = {
        **_BASE_HALLUCINATION_RECORD,
        "explanation": "Second explanation dropped.",
    }
    adjacent = {
        "response_begin": 31,
        "response_end": 65,
        "response_text": "Green bumble fish are also yellow.",
        "faithfulness": "unfaithful",
        "explanation": "Unsupported.",
    }
    parsed = hallucination_contract.parse(json.dumps([first, duplicate, adjacent]))
    assert parsed["items"] == [first, adjacent]


# ---------------------------------------------------------------------------
# DecodeSentences bounds and type checks (issue #1719)
# ---------------------------------------------------------------------------

_CITATIONS_CONFIG_DICT: dict = {
    "model": None,
    "response_format": None,
    "instruction": "Split the last assistant response into individual sentences.",
    "parameters": None,
    "sentence_boundaries": {"documents": "c", "last_message": "r"},
    "docs_as_message": None,
    "transformations": [
        {"type": "explode", "input_path": [], "target_field": "c"},
        {
            "type": "decode_sentences",
            "input_path": [None, "r"],
            "source": "last_message",
            "output_names": {
                "begin": "response_begin",
                "end": "response_end",
                "text": "response_text",
            },
        },
        {
            "type": "decode_sentences",
            "input_path": [None, "c"],
            "source": "documents",
            "output_names": {
                "document_id": "citation_doc_id",
                "begin": "citation_begin",
                "end": "citation_end",
                "text": "citation_text",
            },
        },
        {
            "type": "project",
            "input_path": [None],
            "retained_fields": [
                "response_begin",
                "response_end",
                "response_text",
                "citation_doc_id",
                "citation_begin",
                "citation_end",
                "citation_text",
            ],
        },
        {
            "type": "drop_duplicates",
            "input_path": [],
            "target_fields": ["response_begin", "citation_doc_id", "citation_begin"],
        },
    ],
}

_CONTEXT_ATTRIBUTION_CONFIG_DICT: dict = {
    "model": None,
    "response_format": None,
    "instruction": "You provided the last assistant response above based on context.",
    "parameters": None,
    "sentence_boundaries": {
        "documents": "c",
        "all_but_last_message": "c",
        "last_message": "r",
    },
    "docs_as_message": None,
    "transformations": [
        {"type": "explode", "input_path": [], "target_field": "c"},
        {
            "type": "decode_sentences",
            "input_path": [None, "r"],
            "source": "last_message",
            "output_names": {
                "begin": "response_begin",
                "end": "response_end",
                "text": "response_text",
            },
        },
        {
            "type": "decode_sentences",
            "input_path": [None, "c"],
            "source": ["documents", "all_but_last_message"],
            "output_names": {
                "document_id": "attribution_doc_id",
                "message_index": "attribution_msg_index",
                "begin": "attribution_begin",
                "end": "attribution_end",
                "text": "attribution_text",
            },
        },
        {
            "type": "project",
            "input_path": [None],
            "retained_fields": [
                "response_begin",
                "response_end",
                "response_text",
                "attribution_doc_id",
                "attribution_msg_index",
                "attribution_begin",
                "attribution_end",
                "attribution_text",
            ],
        },
        {
            "type": "drop_duplicates",
            "input_path": [],
            "target_fields": [
                "response_begin",
                "attribution_doc_id",
                "attribution_msg_index",
                "attribution_begin",
            ],
        },
    ],
}


def _make_decode_sentences_rule() -> tuple[DecodeSentences, dict]:
    rule = DecodeSentences(
        {"docs_as_message": None},
        [None, "r"],
        source="last_message",
        output_names={
            "begin": "response_begin",
            "end": "response_end",
            "text": "response_text",
        },
    )
    prepare_output = {
        "begins": [0, 31],
        "ends": [31, 65],
        "texts": [
            "Purple bumble fish are yellow. ",
            "Green bumble fish are also yellow.",
        ],
        "document_ids": [None, None],
        "message_indices": [None, None],
    }
    return rule, prepare_output


@pytest.mark.parametrize("bool_index", [True, False])
def test_decode_sentences_bool_index_raises_type_error(bool_index):
    rule, prepare_output = _make_decode_sentences_rule()
    with pytest.raises(TypeError, match="Expected integer sentence number"):
        rule._transform(bool_index, (0, "r"), prepare_output)


@pytest.mark.parametrize("bad_index", [-1, -2, 2, 5])
def test_decode_sentences_out_of_range_index_raises_value_error(bad_index):
    rule, prepare_output = _make_decode_sentences_rule()
    with pytest.raises(ValueError) as exc_info:
        rule._transform(bad_index, (0, "r"), prepare_output)
    msg = str(exc_info.value)
    assert str(bad_index) in msg
    assert "2 sentence(s)" in msg


def test_decode_sentences_valid_indices_decode_unchanged_citations():
    processor = IntrinsicsResultProcessor(config_dict=_CITATIONS_CONFIG_DICT)
    canned_in = ChatCompletion.model_validate_json(
        (_GRANITE_TESTDATA / "test_canned_input" / "citations.json").read_text(
            encoding="utf-8"
        )
    )
    model_out = ChatCompletionResponse.model_validate_json(
        (
            _GRANITE_TESTDATA / "test_canned_output" / "model_output" / "citations.json"
        ).read_text(encoding="utf-8")
    )
    expected = ChatCompletionResponse.model_validate_json(
        (
            _GRANITE_TESTDATA
            / "test_canned_output"
            / "expected_result"
            / "citations.json"
        ).read_text(encoding="utf-8")
    )

    transformed = processor.transform(model_out, canned_in)
    raw_content = transformed.choices[0].message.content
    assert raw_content is not None
    assert json.loads(raw_content) == json.loads(
        expected.choices[0].message.content or "[]"
    )
    parsed = get_io_contract("citations").parse(raw_content)
    assert len(parsed["items"]) == 3  # type: ignore[arg-type]


def test_decode_sentences_valid_indices_decode_unchanged_context_attribution():
    processor = IntrinsicsResultProcessor(config_dict=_CONTEXT_ATTRIBUTION_CONFIG_DICT)
    canned_in = ChatCompletion.model_validate_json(
        (
            _GRANITE_TESTDATA / "test_canned_input" / "context-attribution.json"
        ).read_text(encoding="utf-8")
    )
    model_out = ChatCompletionResponse.model_validate_json(
        (
            _GRANITE_TESTDATA
            / "test_canned_output"
            / "model_output"
            / "context-attribution.json"
        ).read_text(encoding="utf-8")
    )
    expected = ChatCompletionResponse.model_validate_json(
        (
            _GRANITE_TESTDATA
            / "test_canned_output"
            / "expected_result"
            / "context-attribution.json"
        ).read_text(encoding="utf-8")
    )

    transformed = processor.transform(model_out, canned_in)
    raw_content = transformed.choices[0].message.content
    assert raw_content is not None
    assert json.loads(raw_content) == json.loads(
        expected.choices[0].message.content or "[]"
    )
    parsed = get_io_contract("context-attribution").parse(raw_content)
    assert len(parsed["items"]) == 5  # type: ignore[arg-type]
