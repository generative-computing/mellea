# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wiring tests for check_certainty and find_context_attributions (issue #1516).

Both helpers moved from raw `json.loads` to registry-contract validation when
`call_intrinsic` started parsing via the resolved adapter's contract, and
`find_context_attributions` additionally changed return shape (unwrap of
`"items"`). Their other tests — GPU-gated qualitative/xfail tests in
`test_core.py` plus opt-in e2e runs in `docs/examples` — give no CI signal on
that wiring, so these stub `mfuncs.act` and `backend.resolve_adapter` (the
`test_core_schema.py` pattern) and exercise the helper boundary itself.
"""

import json
import pathlib
from unittest.mock import MagicMock

import pytest

from mellea.backends.adapters import AdapterSchemaMismatchError, get_io_contract
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import _util, core
from mellea.stdlib.context import ChatContext


def _stub_model_call(raw_output: str, monkeypatch, intrinsic_name: str) -> MagicMock:
    """Replace `mfuncs.act` to return *raw_output* and stub `resolve_adapter`
    to carry *intrinsic_name*'s registry contract. Returns the backend mock."""
    backend = MagicMock()
    backend.resolve_adapter.return_value = MagicMock(
        io_contract=get_io_contract(intrinsic_name)
    )

    def fake_act(_intrinsic, context, _backend, *, model_options=None, **_kwargs):
        thunk = MagicMock()
        thunk.is_computed.return_value = True
        thunk.value = raw_output
        return thunk, context

    monkeypatch.setattr(_util.mfuncs, "act", fake_act)
    return backend


def test_check_certainty_resolves_and_parses_via_registry_contract(monkeypatch):
    """check_certainty must resolve `uncertainty` and parse its contract output."""
    backend = _stub_model_call(
        json.dumps({"certainty": 0.9}), monkeypatch, "uncertainty"
    )
    context = ChatContext().add(Message("user", "hi"))

    assert core.check_certainty(context, backend) == pytest.approx(0.9)
    backend.resolve_adapter.assert_called_once_with("uncertainty")


def test_check_certainty_missing_certainty_key_raises(monkeypatch):
    """A missing `certainty` key must raise AdapterSchemaMismatchError."""
    backend = _stub_model_call(
        json.dumps({"wrong_key": 0.9}), monkeypatch, "uncertainty"
    )
    context = ChatContext().add(Message("user", "hi"))

    with pytest.raises(AdapterSchemaMismatchError):
        core.check_certainty(context, backend)


def test_check_certainty_rejects_non_object_output(monkeypatch):
    """A top-level JSON array is a ValueError, not a schema mismatch."""
    backend = _stub_model_call(json.dumps([0.9]), monkeypatch, "uncertainty")
    context = ChatContext().add(Message("user", "hi"))

    with pytest.raises(ValueError, match="must be a JSON object"):
        core.check_certainty(context, backend)


def test_find_context_attributions_returns_items_from_recorded_output(monkeypatch):
    """The `result_json["items"]` unwrap must survive the helper boundary."""
    fixture = (
        pathlib.Path(__file__).resolve().parent
        / "testdata"
        / "output_json"
        / "context-attribution.json"
    )
    completion = json.loads(fixture.read_text(encoding="utf-8"))
    raw = completion["choices"][0]["message"]["content"]

    backend = _stub_model_call(raw, monkeypatch, "context-attribution")
    context = ChatContext().add(Message("user", "hi"))

    result = core.find_context_attributions(
        "The answer is 42.", ["A document."], context, backend
    )

    assert len(result) == 7
    assert result[0]["attribution_msg_index"] is None
    backend.resolve_adapter.assert_called_once_with("context-attribution")


def test_find_context_attributions_rejects_non_array_output(monkeypatch):
    """A top-level JSON object is a ValueError for the list contract."""
    backend = _stub_model_call(
        json.dumps({"not": "a list"}), monkeypatch, "context-attribution"
    )
    context = ChatContext().add(Message("user", "hi"))

    with pytest.raises(ValueError, match="must be a JSON array"):
        core.find_context_attributions(
            "The answer is 42.", ["A document."], context, backend
        )
