# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the guardian adapter functions' output contracts (Epic #929,
#1516).

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
import unittest.mock

import pytest

from mellea.backends.adapters import (
    AdapterMixin,
    AdapterSchemaMismatchError,
    get_io_contract,
)
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext

# ---------------------------------------------------------------------------
# policy_guardrails
# ---------------------------------------------------------------------------


def test_policy_guardrails_contract_enforced_neither_key() -> None:
    contract = get_io_contract("policy-guardrails")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"wrong_key": "value"}))
    err = exc_info.value
    assert err.name == "policy-guardrails"
    assert "label" in err.expected_keys
    assert "score" in err.expected_keys


def test_policy_guardrails_contract_enforced_both_keys() -> None:
    contract = get_io_contract("policy-guardrails")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"label": "Yes", "score": "Yes"}))
    err = exc_info.value
    assert err.name == "policy-guardrails"


def test_policy_guardrails_forward_compat_label() -> None:
    contract = get_io_contract("policy-guardrails")
    result = contract.parse(json.dumps({"label": "Yes", "extra": "ignored"}))
    assert result["label"] == "Yes"


def test_policy_guardrails_forward_compat_score() -> None:
    contract = get_io_contract("policy-guardrails")
    result = contract.parse(json.dumps({"score": "No", "extra": "ignored"}))
    assert result["score"] == "No"


def test_policy_guardrails_rejects_non_dict() -> None:
    contract = get_io_contract("policy-guardrails")
    with pytest.raises(ValueError, match="must be a JSON object"):
        contract.parse(json.dumps(["not", "a", "dict"]))


# ---------------------------------------------------------------------------
# guardian_check
# ---------------------------------------------------------------------------


def test_guardian_check_contract_enforced_missing_guardian_key() -> None:
    contract = get_io_contract("guardian-core")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"wrong_key": 0.5}))
    err = exc_info.value
    assert err.name == "guardian-core"
    assert "guardian" in err.expected_keys


def test_guardian_check_contract_enforced_missing_score_in_guardian() -> None:
    contract = get_io_contract("guardian-core")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"guardian": {"wrong_key": 0.5}}))
    err = exc_info.value
    assert err.name == "guardian-core"
    assert "score" in err.expected_keys


def test_guardian_check_contract_enforced_guardian_not_dict() -> None:
    contract = get_io_contract("guardian-core")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"guardian": 0.8}))
    err = exc_info.value
    assert err.name == "guardian-core"
    assert "score" in err.expected_keys
    assert "guardian" in err.observed_keys


def test_guardian_check_forward_compat() -> None:
    contract = get_io_contract("guardian-core")
    result = contract.parse(
        json.dumps({"guardian": {"score": 0.9}, "extra": "ignored"})
    )
    assert isinstance(result["guardian"], dict)
    assert result["guardian"]["score"] == 0.9  # type: ignore[index]


def test_guardian_check_rejects_non_dict() -> None:
    contract = get_io_contract("guardian-core")
    with pytest.raises(ValueError, match="must be a JSON object"):
        contract.parse(json.dumps([0.5]))


# ---------------------------------------------------------------------------
# factuality_detection
# ---------------------------------------------------------------------------


def test_factuality_detection_contract_enforced() -> None:
    contract = get_io_contract("factuality-detection")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"wrong_key": "yes"}))
    err = exc_info.value
    assert err.name == "factuality-detection"
    assert "score" in err.expected_keys


def test_factuality_detection_forward_compat() -> None:
    contract = get_io_contract("factuality-detection")
    result = contract.parse(json.dumps({"score": "yes", "confidence": 0.9}))
    assert result["score"] == "yes"


def test_factuality_detection_rejects_non_dict() -> None:
    contract = get_io_contract("factuality-detection")
    with pytest.raises(ValueError, match="must be a JSON object"):
        contract.parse(json.dumps("yes"))


# ---------------------------------------------------------------------------
# factuality_correction
# ---------------------------------------------------------------------------


def test_factuality_correction_contract_enforced() -> None:
    contract = get_io_contract("factuality-correction")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({"wrong_key": "corrected text"}))
    err = exc_info.value
    assert err.name == "factuality-correction"
    assert "correction" in err.expected_keys


def test_factuality_correction_forward_compat() -> None:
    contract = get_io_contract("factuality-correction")
    result = contract.parse(
        json.dumps({"correction": "The correct answer is 42.", "score": 0.95})
    )
    assert result["correction"] == "The correct answer is 42."


def test_factuality_correction_rejects_non_dict() -> None:
    contract = get_io_contract("factuality-correction")
    with pytest.raises(ValueError, match="must be a JSON object"):
        contract.parse(json.dumps(["not", "a", "dict"]))


# ---------------------------------------------------------------------------
# Error message includes adapter name for debuggability
# ---------------------------------------------------------------------------


def test_policy_guardrails_error_mentions_adapter_name() -> None:
    contract = get_io_contract("policy-guardrails")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({}))
    assert exc_info.value.name == "policy-guardrails"


def test_guardian_check_error_mentions_adapter_name() -> None:
    contract = get_io_contract("guardian-core")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({}))
    assert exc_info.value.name == "guardian-core"


def test_factuality_detection_error_mentions_adapter_name() -> None:
    contract = get_io_contract("factuality-detection")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({}))
    assert exc_info.value.name == "factuality-detection"


def test_factuality_correction_error_mentions_adapter_name() -> None:
    contract = get_io_contract("factuality-correction")
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        contract.parse(json.dumps({}))
    assert exc_info.value.name == "factuality-correction"


# ---------------------------------------------------------------------------
# policy_guardrails helper — score branch
# ---------------------------------------------------------------------------


def test_policy_guardrails_score_branch(monkeypatch) -> None:
    """policy_guardrails returns the `score` value when the adapter omits `label`."""

    def fake_call_intrinsic(name, context, backend, /, kwargs=None, model_options=None):
        return {"score": "No"}

    monkeypatch.setattr(guardian, "call_intrinsic", fake_call_intrinsic)
    result = guardian.policy_guardrails(
        ChatContext(),
        unittest.mock.create_autospec(AdapterMixin, instance=True),
        policy_text="any policy",
    )
    assert result == "No"
