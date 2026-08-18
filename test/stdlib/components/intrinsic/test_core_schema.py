# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration-style unit tests for `core.requirement_check`'s schema validation.

Issue #1516 replaced the hand-rolled score-range validation `requirement_check` used
to run after `call_intrinsic` with the `requirement-check` capability's declared
`IOContract`, obtained from the adapter `resolve_adapter` returns rather than passed
as a parallel argument. These tests exercise that real path — `call_intrinsic`,
`resolve_adapter`, and `IOContract.parse` all run — by mocking only `mfuncs.act` (the
actual model call), not `call_intrinsic` itself.

Score-range edge cases (NaN, bool, out-of-range, ...) are covered directly against
the contract in `test/backends/test_adapters/test_io_contracts.py`; this file checks
that `core.requirement_check` is wired to that contract end-to-end.
"""

import json
from unittest.mock import MagicMock

import pytest

from mellea.backends.adapters import AdapterSchemaMismatchError
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import _util, core
from mellea.stdlib.components.intrinsic.core import _REQUIREMENT_CHECK_ADAPTER
from mellea.stdlib.context import ChatContext

_REQUIREMENT = "must be polite"


def _call(result_dict: dict, monkeypatch) -> float:
    backend = MagicMock()
    backend.resolve_adapter.return_value = _REQUIREMENT_CHECK_ADAPTER

    def fake_act(_intrinsic, context, _backend, *, model_options=None, **_kwargs):
        thunk = MagicMock()
        thunk.is_computed.return_value = True
        thunk.value = json.dumps(result_dict)
        return thunk, context

    monkeypatch.setattr(_util.mfuncs, "act", fake_act)
    context = ChatContext().add(Message("user", "hi"))
    return core.requirement_check(context, backend, _REQUIREMENT)


def test_valid_score_returned(monkeypatch):
    assert _call({"requirement_check": {"score": 0.8}}, monkeypatch) == pytest.approx(
        0.8
    )


def test_boundary_score_zero(monkeypatch):
    assert _call({"requirement_check": {"score": 0.0}}, monkeypatch) == pytest.approx(
        0.0
    )


def test_boundary_score_one(monkeypatch):
    assert _call({"requirement_check": {"score": 1.0}}, monkeypatch) == pytest.approx(
        1.0
    )


def test_missing_requirement_check_key_raises(monkeypatch):
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"other_field": 0.9}, monkeypatch)


def test_missing_score_key_raises(monkeypatch):
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"other_key": 0.9}}, monkeypatch)


def test_score_above_range_raises(monkeypatch):
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"score": 1.5}}, monkeypatch)


def test_score_below_range_raises(monkeypatch):
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"score": -0.1}}, monkeypatch)


def test_requirement_check_resolves_adapter_by_name(monkeypatch):
    """core.requirement_check must resolve the `requirement-check` capability."""
    backend = MagicMock()
    backend.resolve_adapter.return_value = _REQUIREMENT_CHECK_ADAPTER

    def fake_act(_intrinsic, context, _backend, *, model_options=None, **_kwargs):
        thunk = MagicMock()
        thunk.is_computed.return_value = True
        thunk.value = json.dumps({"requirement_check": {"score": 0.5}})
        return thunk, context

    monkeypatch.setattr(_util.mfuncs, "act", fake_act)
    context = ChatContext().add(Message("user", "hi"))
    core.requirement_check(context, backend, _REQUIREMENT)
    backend.resolve_adapter.assert_called_once_with("requirement-check")
