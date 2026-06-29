"""Unit tests for IOContract validation in guardian.py (Epic #929 Phase 1).

Tests the `parse()` method of each IOContract subclass directly — no backend,
no GPU, no model download required.  Two tests per helper:

- `test_<helper>_contract_enforced` — output missing a required field raises
  :class:`~mellea.backends.adapters.AdapterSchemaMismatchError`.
- `test_<helper>_forward_compat` — output containing an extra optional field
  does *not* raise.
"""

import json

import pytest

from mellea.backends.adapters import AdapterSchemaMismatchError
from mellea.stdlib.components.intrinsic.guardian import (
    _FACTUALITY_CORRECTION_ADAPTER,
    _FACTUALITY_DETECTION_ADAPTER,
    _GUARDIAN_CHECK_ADAPTER,
    _POLICY_GUARDRAILS_ADAPTER,
)

# ---------------------------------------------------------------------------
# policy_guardrails
# ---------------------------------------------------------------------------


def test_policy_guardrails_contract_enforced_neither_key() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _POLICY_GUARDRAILS_ADAPTER.io_contract.parse(json.dumps({"wrong_key": "value"}))
    err = exc_info.value
    assert err.name == "policy-guardrails"
    assert "label" in err.expected_keys
    assert "score" in err.expected_keys


def test_policy_guardrails_contract_enforced_both_keys() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _POLICY_GUARDRAILS_ADAPTER.io_contract.parse(
            json.dumps({"label": "Yes", "score": "Yes"})
        )
    err = exc_info.value
    assert err.name == "policy-guardrails"


def test_policy_guardrails_forward_compat_label() -> None:
    result = _POLICY_GUARDRAILS_ADAPTER.io_contract.parse(
        json.dumps({"label": "Yes", "extra": "ignored"})
    )
    assert result["label"] == "Yes"


def test_policy_guardrails_forward_compat_score() -> None:
    result = _POLICY_GUARDRAILS_ADAPTER.io_contract.parse(
        json.dumps({"score": "No", "extra": "ignored"})
    )
    assert result["score"] == "No"


def test_policy_guardrails_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        _POLICY_GUARDRAILS_ADAPTER.io_contract.parse(json.dumps(["not", "a", "dict"]))


# ---------------------------------------------------------------------------
# guardian_check
# ---------------------------------------------------------------------------


def test_guardian_check_contract_enforced_missing_guardian_key() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _GUARDIAN_CHECK_ADAPTER.io_contract.parse(json.dumps({"wrong_key": 0.5}))
    err = exc_info.value
    assert err.name == "guardian-core"
    assert "guardian" in err.expected_keys


def test_guardian_check_contract_enforced_missing_score_in_guardian() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _GUARDIAN_CHECK_ADAPTER.io_contract.parse(
            json.dumps({"guardian": {"wrong_key": 0.5}})
        )
    err = exc_info.value
    assert err.name == "guardian-core"
    assert "score" in err.expected_keys


def test_guardian_check_contract_enforced_guardian_not_dict() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _GUARDIAN_CHECK_ADAPTER.io_contract.parse(json.dumps({"guardian": 0.8}))
    err = exc_info.value
    assert err.name == "guardian-core"
    assert "score" in err.expected_keys


def test_guardian_check_forward_compat() -> None:
    result = _GUARDIAN_CHECK_ADAPTER.io_contract.parse(
        json.dumps({"guardian": {"score": 0.9}, "extra": "ignored"})
    )
    assert isinstance(result["guardian"], dict)
    assert result["guardian"]["score"] == 0.9  # type: ignore[index]


def test_guardian_check_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        _GUARDIAN_CHECK_ADAPTER.io_contract.parse(json.dumps([0.5]))


# ---------------------------------------------------------------------------
# factuality_detection
# ---------------------------------------------------------------------------


def test_factuality_detection_contract_enforced() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _FACTUALITY_DETECTION_ADAPTER.io_contract.parse(
            json.dumps({"wrong_key": "yes"})
        )
    err = exc_info.value
    assert err.name == "factuality-detection"
    assert "score" in err.expected_keys


def test_factuality_detection_forward_compat() -> None:
    result = _FACTUALITY_DETECTION_ADAPTER.io_contract.parse(
        json.dumps({"score": "yes", "confidence": 0.9})
    )
    assert result["score"] == "yes"


def test_factuality_detection_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        _FACTUALITY_DETECTION_ADAPTER.io_contract.parse(json.dumps("yes"))


# ---------------------------------------------------------------------------
# factuality_correction
# ---------------------------------------------------------------------------


def test_factuality_correction_contract_enforced() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _FACTUALITY_CORRECTION_ADAPTER.io_contract.parse(
            json.dumps({"wrong_key": "corrected text"})
        )
    err = exc_info.value
    assert err.name == "factuality-correction"
    assert "correction" in err.expected_keys


def test_factuality_correction_forward_compat() -> None:
    result = _FACTUALITY_CORRECTION_ADAPTER.io_contract.parse(
        json.dumps({"correction": "The correct answer is 42.", "score": 0.95})
    )
    assert result["correction"] == "The correct answer is 42."


def test_factuality_correction_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        _FACTUALITY_CORRECTION_ADAPTER.io_contract.parse(
            json.dumps(["not", "a", "dict"])
        )


# ---------------------------------------------------------------------------
# Error message includes adapter name for debuggability
# ---------------------------------------------------------------------------


def test_policy_guardrails_error_mentions_adapter_name() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _POLICY_GUARDRAILS_ADAPTER.io_contract.parse(json.dumps({}))
    assert exc_info.value.name == "policy-guardrails"


def test_guardian_check_error_mentions_adapter_name() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _GUARDIAN_CHECK_ADAPTER.io_contract.parse(json.dumps({}))
    assert exc_info.value.name == "guardian-core"


def test_factuality_detection_error_mentions_adapter_name() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _FACTUALITY_DETECTION_ADAPTER.io_contract.parse(json.dumps({}))
    assert exc_info.value.name == "factuality-detection"


def test_factuality_correction_error_mentions_adapter_name() -> None:
    with pytest.raises(AdapterSchemaMismatchError) as exc_info:
        _FACTUALITY_CORRECTION_ADAPTER.io_contract.parse(json.dumps({}))
    assert exc_info.value.name == "factuality-correction"
