# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `call_intrinsic`'s model_options resolution.

Exercises the model_options precedence without a real backend or model —
guards against the PR #972 bug class (caller-supplied model_options silently
discarded behind a hardcoded default) resurfacing.
"""

import json
from unittest.mock import MagicMock

from mellea.backends.model_options import ModelOption
from mellea.stdlib.components.intrinsic import _util
from mellea.stdlib.context import ChatContext


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
    context = ChatContext()

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
    context = ChatContext()

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
    context = ChatContext()

    _util.call_intrinsic("answerability", context, backend)

    backend.resolve_adapter.assert_called_once_with("answerability")
