# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the real AdapterMixin.adapter_scope implementation (issue #1141).

Exercises activate/deactivate ordering and exception-safety using a fake
WeightsBinding double — no real backend or model required. `adapter_scope` fires
metric hooks only and opens no spans, so no exporter is involved; the hook
dispatch safely no-ops when no plugins are registered.
"""

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from mellea.backends.adapters._core import (
    Adapter,
    AdapterSchemaMismatchError,
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


@contextlib.contextmanager
def _capture_hooks():
    """Capture fired hook payloads without a live plugin manager.

    Follows the idiom in `test_local_file_binding.py`: patch `has_plugins` on and
    `_run_async_in_thread` off, with `invoke_hook` a plain `MagicMock` so no
    coroutine is created (a real one would trigger "never awaited" warnings, since
    the dispatch path is not live in tests).
    """
    with (
        patch("mellea.backends.adapters.adapter.has_plugins", return_value=True),
        patch("mellea.backends.adapters.adapter.invoke_hook") as mock_invoke,
        patch("mellea.backends.adapters.adapter._run_async_in_thread"),
    ):
        yield mock_invoke


def _outcomes(mock_invoke):
    payloads = [c.args[1] for c in mock_invoke.call_args_list]
    return [p for p in payloads if hasattr(p, "outcome")]


def test_adapter_scope_reports_schema_mismatch_as_schema_error():
    """An AdapterSchemaMismatchError is `schema_error`, not a generic `error`.

    `mellea.adapter_function.parse_failures` increments only on `schema_error`, so
    collapsing this into `error` would leave that counter permanently at zero.
    """
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, _ = _make_adapter()

    with _capture_hooks() as mock_invoke:
        with pytest.raises(AdapterSchemaMismatchError):
            with AdapterMixin.adapter_scope(mock_backend, adapter):
                raise AdapterSchemaMismatchError(
                    "answerability",
                    frozenset({"wrong_key"}),
                    frozenset({"answerability"}),
                )

    invocations = _outcomes(mock_invoke)
    assert [p.outcome for p in invocations] == ["schema_error"]
    assert isinstance(invocations[0].error, AdapterSchemaMismatchError)


def test_adapter_scope_reports_other_exceptions_as_error():
    """Anything that is not a schema mismatch stays `error`."""
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, _ = _make_adapter()

    with _capture_hooks() as mock_invoke:
        with pytest.raises(RuntimeError, match="boom"):
            with AdapterMixin.adapter_scope(mock_backend, adapter):
                raise RuntimeError("boom")

    assert [p.outcome for p in _outcomes(mock_invoke)] == ["error"]


def test_adapter_scope_noop_when_adapter_is_none():
    mock_backend = MagicMock(spec=AdapterMixin)

    entered = False
    with AdapterMixin.adapter_scope(mock_backend, None):
        entered = True

    assert entered


if __name__ == "__main__":
    pytest.main([__file__])
