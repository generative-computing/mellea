# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `call_intrinsic`'s model_options resolution and contract
wiring.

Exercises the model_options precedence without a real backend or model —
guards against the PR #972 bug class (caller-supplied model_options silently
discarded behind a hardcoded default) resurfacing. Also covers the issue #1516
change: the output contract is taken from the adapter `resolve_adapter()`
returns, and `call_intrinsic` no longer accepts an `io_contract` kwarg.
"""

import json
from unittest.mock import MagicMock

import pytest

from mellea.backends.model_options import ModelOption
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import _util
from mellea.stdlib.context import ChatContext
from mellea.stdlib.context.simple import SimpleContext


def _fake_act_capturing(calls):
    def _act(_intrinsic, context, _backend, *, model_options=None, **_kwargs):
        calls.append(model_options)
        thunk = MagicMock()
        thunk.is_computed.return_value = True
        thunk.value = json.dumps({"result": "ok"})
        return thunk, context

    return _act


def test_call_intrinsic_caller_model_options_survive(monkeypatch):
    """Caller-supplied model_options must not be clobbered by the temperature default."""
    calls: list[dict | None] = []
    monkeypatch.setattr(_util.mfuncs, "act", _fake_act_capturing(calls))

    backend = MagicMock()
    context = ChatContext().add(Message("user", "hi"))

    _util.call_intrinsic(
        "answerability", context, backend, model_options={ModelOption.TEMPERATURE: 0.7}
    )

    assert len(calls) == 1
    resolved = calls[0]
    assert resolved is not None
    assert resolved[ModelOption.TEMPERATURE] == 0.7


def test_call_intrinsic_default_temperature_used_when_not_overridden(monkeypatch):
    """When the caller passes no model_options, the 0.0 default is used."""
    calls: list[dict | None] = []
    monkeypatch.setattr(_util.mfuncs, "act", _fake_act_capturing(calls))

    backend = MagicMock()
    context = ChatContext().add(Message("user", "hi"))

    _util.call_intrinsic("answerability", context, backend)

    assert len(calls) == 1
    resolved = calls[0]
    assert resolved is not None
    assert resolved[ModelOption.TEMPERATURE] == 0.0


def test_call_intrinsic_resolves_adapter_before_acting(monkeypatch):
    """resolve_adapter is called to register/lazily create the adapter."""
    calls: list[dict | None] = []
    monkeypatch.setattr(_util.mfuncs, "act", _fake_act_capturing(calls))

    backend = MagicMock()
    context = ChatContext().add(Message("user", "hi"))

    _util.call_intrinsic("answerability", context, backend)

    backend.resolve_adapter.assert_called_once_with("answerability")


def test_call_intrinsic_rejects_context_with_no_history(monkeypatch):
    """A SimpleContext forwards no history, so call_intrinsic must fail early (issue #937)."""
    calls: list[dict | None] = []
    monkeypatch.setattr(_util.mfuncs, "act", _fake_act_capturing(calls))

    backend = MagicMock()
    # SimpleContext.view_for_generation() is always [] even after add().
    context = SimpleContext().add(Message("user", "hi"))

    with pytest.raises(ValueError, match="forwards no history"):
        _util.call_intrinsic("answerability", context, backend)

    # The guard must fire before the backend is touched.
    backend.resolve_adapter.assert_not_called()
    assert calls == []


def test_call_intrinsic_rejects_empty_chat_context(monkeypatch):
    """An empty ChatContext also forwards nothing and must be rejected."""
    calls: list[dict | None] = []
    monkeypatch.setattr(_util.mfuncs, "act", _fake_act_capturing(calls))

    backend = MagicMock()

    with pytest.raises(ValueError, match="forwards no history"):
        _util.call_intrinsic("answerability", ChatContext(), backend)

    backend.resolve_adapter.assert_not_called()
    assert calls == []


def test_call_intrinsic_parses_via_resolved_adapters_io_contract(monkeypatch):
    """The output contract must come from resolve_adapter's return value, not a
    parallel argument (issue #1516)."""
    calls: list[dict | None] = []
    monkeypatch.setattr(_util.mfuncs, "act", _fake_act_capturing(calls))

    resolved_adapter = MagicMock()
    resolved_adapter.io_contract.parse.return_value = {"parsed": "by-resolved-adapter"}
    backend = MagicMock()
    backend.resolve_adapter.return_value = resolved_adapter
    context = ChatContext().add(Message("user", "hi"))

    result = _util.call_intrinsic("answerability", context, backend)

    resolved_adapter.io_contract.parse.assert_called_once_with(
        json.dumps({"result": "ok"})
    )
    assert result == {"parsed": "by-resolved-adapter"}


def test_call_intrinsic_no_longer_accepts_io_contract_kwarg(monkeypatch):
    """A caller can no longer pass a separate, possibly-mismatched io_contract."""
    calls: list[dict | None] = []
    monkeypatch.setattr(_util.mfuncs, "act", _fake_act_capturing(calls))

    backend = MagicMock()
    context = ChatContext().add(Message("user", "hi"))

    with pytest.raises(TypeError, match="io_contract"):
        _util.call_intrinsic(
            "answerability",
            context,
            backend,
            io_contract=MagicMock(),  # type: ignore[call-arg]
        )
