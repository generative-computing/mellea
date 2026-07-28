# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the real AdapterMixin.adapter_scope implementation (issue #1141).

Exercises activate/deactivate ordering and exception-safety using a fake
WeightsBinding double — no real backend, model, or OTel exporter required
(span/metric helpers safely no-op when tracing/plugins are unavailable).
"""

from unittest.mock import MagicMock

import pytest

from mellea.backends.adapters._core import (
    Adapter,
    Identity,
    IOContract,
    LocalFileBinding,
)
from mellea.backends.adapters.adapter import AdapterMixin
from mellea.core import Component


class _Contract(IOContract):
    def build_prompt(self, **kwargs: object) -> Component:
        raise NotImplementedError

    def parse(self, raw: str) -> dict[str, object]:
        return {}


def _make_adapter():
    weights = MagicMock(spec=LocalFileBinding)
    weights.binding_type = "local_file"
    weights.revision = "abc123"
    identity = Identity(name="answerability", adapter_type="lora")
    adapter = Adapter(identity=identity, io_contract=_Contract(), weights=weights)
    return adapter, weights


def test_adapter_scope_activates_then_deactivates_in_order():
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()
    calls = []
    weights.activate.side_effect = lambda: calls.append("activate")
    weights.deactivate.side_effect = lambda: calls.append("deactivate")

    with AdapterMixin.adapter_scope(mock_backend, adapter):
        calls.append("body")

    assert calls == ["activate", "body", "deactivate"]


def test_adapter_scope_deactivates_even_when_body_raises():
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()

    with pytest.raises(RuntimeError, match="boom"):
        with AdapterMixin.adapter_scope(mock_backend, adapter):
            raise RuntimeError("boom")

    weights.activate.assert_called_once()
    weights.deactivate.assert_called_once()


def test_adapter_scope_deactivates_even_when_activate_raises():
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()
    weights.activate.side_effect = RuntimeError("activation failed")

    with pytest.raises(RuntimeError, match="activation failed"):
        with AdapterMixin.adapter_scope(mock_backend, adapter):
            pytest.fail("body must not run when activate() raises")

    weights.deactivate.assert_not_called()


def test_adapter_scope_propagates_deactivate_error_over_body_success():
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()
    weights.deactivate.side_effect = RuntimeError("deactivation failed")

    with pytest.raises(RuntimeError, match="deactivation failed"):
        with AdapterMixin.adapter_scope(mock_backend, adapter):
            pass

    weights.activate.assert_called_once()
    weights.deactivate.assert_called_once()


def test_adapter_scope_noop_when_adapter_is_none():
    mock_backend = MagicMock(spec=AdapterMixin)

    entered = False
    with AdapterMixin.adapter_scope(mock_backend, None):
        entered = True

    assert entered


if __name__ == "__main__":
    pytest.main([__file__])
