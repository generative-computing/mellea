# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the canonical adapter-function output-contract registry (issue #1516).

Covers the registry's completeness/exhaustiveness invariant, the fallback contract
for names outside the catalog, and the `_RequirementCheckContract` parsing logic that
replaced `core.requirement_check`'s hand-rolled post-call validation.
"""

import json
import math

import pytest

from mellea.backends.adapters import AdapterSchemaMismatchError
from mellea.backends.adapters.catalog import known_intrinsic_names
from mellea.backends.adapters.io_contracts import (
    _INTRINSIC_IO_CONTRACTS,
    get_io_contract,
)

# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_registry_covers_every_catalog_name():
    """Every catalogued adapter function must have a declared contract.

    Regression guard for the acceptance criterion that no adapter reachable
    through `resolve_adapter` carries an unimplemented placeholder contract.
    """
    missing = set(known_intrinsic_names()) - set(_INTRINSIC_IO_CONTRACTS)
    assert missing == set()


def test_get_io_contract_returns_declared_instance_for_known_names():
    for name in known_intrinsic_names():
        assert get_io_contract(name) is _INTRINSIC_IO_CONTRACTS[name]


def test_get_io_contract_falls_back_permissively_for_unknown_names():
    """A name outside the catalog (e.g. a `CustomIntrinsicAdapter`) must not raise."""
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
