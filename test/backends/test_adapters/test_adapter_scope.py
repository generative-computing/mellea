# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the real AdapterMixin.adapter_scope implementation (issue #1141).

Exercises activate/deactivate ordering and exception-safety using a fake
WeightsBinding double — no real backend or model required. `adapter_scope` fires
metric hooks only and opens no spans, so no exporter is involved; the hook
dispatch safely no-ops when no plugins are registered.
"""

from types import SimpleNamespace
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
from mellea.backends.adapters.catalog import AdapterType
from mellea.core import Component
from test.backends.test_adapters._hook_capture import (
    capture_adapter_hooks,
    invocation_payloads,
)


class _Contract(IOContract):
    def build_prompt(self, **kwargs: object) -> Component:
        raise NotImplementedError

    def parse(self, raw: str) -> dict[str, object]:
        return {}


def _make_adapter():
    weights = MagicMock(spec=LocalFileBinding)
    weights.binding_type = "local_file"
    weights.revision = "abc123"
    # `MagicMock(spec=LocalFileBinding)` passes `isinstance(weights,
    # LocalFileBinding)`, so `adapter_scope` calls `resolved_revision()` on it
    # (real `LocalFileBinding`s resolve lazily); stub it to match `.revision`
    # since these tests aren't exercising lazy resolution itself.
    weights.resolved_revision.return_value = "abc123"
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


def test_adapter_scope_preserves_body_error_when_deactivate_also_raises():
    """The body's exception remains primary when cleanup also fails."""
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()
    body_error = ValueError("body failed")
    deactivate_error = RuntimeError("deactivation failed")
    weights.deactivate.side_effect = deactivate_error

    with capture_adapter_hooks() as mock_invoke:
        with pytest.raises(ValueError, match="body failed") as exc_info:
            with AdapterMixin.adapter_scope(mock_backend, adapter):
                raise body_error

    assert exc_info.value is body_error
    assert exc_info.value.__cause__ is deactivate_error
    weights.deactivate.assert_called_once()
    invocations = invocation_payloads(mock_invoke)
    assert [p.outcome for p in invocations] == ["error"]
    assert invocations[0].error is body_error


def test_adapter_scope_deactivates_even_when_activate_raises():
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()
    weights.activate.side_effect = RuntimeError("activation failed")

    with pytest.raises(RuntimeError, match="activation failed"):
        with AdapterMixin.adapter_scope(mock_backend, adapter):
            pytest.fail("body must not run when activate() raises")

    weights.deactivate.assert_not_called()


@pytest.mark.parametrize("failing_phase", ["activate", "deactivate"])
def test_adapter_scope_ignores_phase_hook_dispatch_failures(failing_phase: str):
    """A phase-hook failure must not break an otherwise successful scope."""
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()
    body_ran = False

    def _raise_on_phase_hook(hook_type: object, payload: object) -> None:
        if getattr(payload, "phase", None) == failing_phase:
            raise RuntimeError("plugin dispatch blew up")

    with (
        patch("mellea.backends.adapters.adapter.has_plugins", return_value=True),
        patch(
            "mellea.plugins.hooks.adapter_function.AdapterFunctionPhaseCompletePayload",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        patch(
            "mellea.backends.adapters.adapter.invoke_hook",
            side_effect=_raise_on_phase_hook,
        ),
    ):
        with AdapterMixin.adapter_scope(mock_backend, adapter):
            body_ran = True

    assert body_ran
    weights.activate.assert_called_once()
    weights.deactivate.assert_called_once()


def test_adapter_scope_propagates_deactivate_error_over_body_success():
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()
    weights.deactivate.side_effect = RuntimeError("deactivation failed")

    with capture_adapter_hooks() as mock_invoke:
        with pytest.raises(RuntimeError, match="deactivation failed"):
            with AdapterMixin.adapter_scope(mock_backend, adapter):
                pass

    weights.activate.assert_called_once()
    weights.deactivate.assert_called_once()

    # A body that succeeded does not make the invocation a success: the failure
    # came from deactivate, and the invocation must still report it.
    invocations = invocation_payloads(mock_invoke)
    assert [p.outcome for p in invocations] == ["error"]
    assert isinstance(invocations[0].error, RuntimeError)


def test_adapter_scope_reports_schema_mismatch_as_schema_error():
    """An AdapterSchemaMismatchError is `schema_error`, not a generic `error`.

    `mellea.adapter_function.parse_failures` increments only on `schema_error`, so
    collapsing this into `error` would leave that counter permanently at zero.
    """
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, _ = _make_adapter()

    with capture_adapter_hooks() as mock_invoke:
        with pytest.raises(AdapterSchemaMismatchError):
            with AdapterMixin.adapter_scope(mock_backend, adapter):
                raise AdapterSchemaMismatchError(
                    "answerability",
                    frozenset({"wrong_key"}),
                    frozenset({"answerability"}),
                )

    invocations = invocation_payloads(mock_invoke)
    assert [p.outcome for p in invocations] == ["schema_error"]
    assert isinstance(invocations[0].error, AdapterSchemaMismatchError)


def test_adapter_scope_reports_other_exceptions_as_error():
    """Anything that is not a schema mismatch stays `error`."""
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, _ = _make_adapter()

    with capture_adapter_hooks() as mock_invoke:
        with pytest.raises(RuntimeError, match="boom"):
            with AdapterMixin.adapter_scope(mock_backend, adapter):
                raise RuntimeError("boom")

    assert [p.outcome for p in invocation_payloads(mock_invoke)] == ["error"]


def test_phase_hook_not_fired_when_the_phase_itself_fails():
    """A phase that raised did not complete, so no phase event is emitted.

    `ADAPTER_FUNCTION_PHASE_COMPLETE` means the phase finished. The failure is
    reported once, at invocation level, where `outcome`/`error` carry it — so a
    consumer reconciling phase counts against invocation counts sees one
    invocation error and no phase event, not both.
    """
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()
    weights.activate.side_effect = RuntimeError("activation failed")

    with capture_adapter_hooks() as mock_invoke:
        with pytest.raises(RuntimeError, match="activation failed"):
            with AdapterMixin.adapter_scope(mock_backend, adapter):
                pytest.fail("body must not run when activate() raises")

    payloads = [c.args[1] for c in mock_invoke.call_args_list]
    assert [p for p in payloads if hasattr(p, "phase")] == []

    invocations = invocation_payloads(mock_invoke)
    assert [p.outcome for p in invocations] == ["error"]


def test_adapter_scope_reports_resolved_revision_not_raw_none():
    """A lazily-resolved binding (revision=None) must report its resolved pin, not None.

    Regression guard: `adapter_scope` used to read the raw `.revision`
    attribute, which is `None` for a `LocalFileBinding(name=..., revision=None)`
    even though the binding downloads and runs against a concrete catalogue
    pin. Reporting `None` mislabels an effectively-pinned invocation as
    unpinned in telemetry.
    """
    mock_backend = MagicMock(spec=AdapterMixin)
    binding = LocalFileBinding(name="answerability")  # revision=None, lazily resolved
    identity = Identity(name="answerability", adapter_type="lora")
    adapter = Adapter(identity=identity, io_contract=_Contract(), weights=binding)
    binding.activate = MagicMock()
    binding.deactivate = MagicMock()
    assert binding.revision is None

    with capture_adapter_hooks() as mock_invoke:
        with AdapterMixin.adapter_scope(mock_backend, adapter):
            pass

    invocations = invocation_payloads(mock_invoke)
    assert len(invocations) == 1
    assert invocations[0].revision == binding.resolved_revision()
    assert invocations[0].revision != "main"


def test_adapter_scope_swallows_invocation_hook_failure_on_clean_run():
    """A failing invocation-complete hook must not turn a clean run into an error.

    Regression guard: `_fire_invocation_complete` used to be called unguarded
    in the outer `finally`. If its hook dispatch raised, that exception
    replaced the (successful, no-exception) outcome of an otherwise-clean
    `with` block — telemetry turning success into failure.
    """
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()

    with patch(
        "mellea.backends.adapters.adapter._fire_invocation_complete",
        side_effect=RuntimeError("invocation hook dispatch blew up"),
    ):
        with AdapterMixin.adapter_scope(mock_backend, adapter):
            pass  # must not raise despite the hook failing on exit

    weights.activate.assert_called_once()
    weights.deactivate.assert_called_once()


def test_adapter_scope_invocation_hook_failure_does_not_mask_body_exception():
    """A failing invocation-complete hook must not replace the body's real exception.

    Regression guard: when both the body and the invocation hook raise, the
    caller must still see the body's exception, not the hook's.
    """
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter, weights = _make_adapter()

    with patch(
        "mellea.backends.adapters.adapter._fire_invocation_complete",
        side_effect=RuntimeError("invocation hook dispatch blew up"),
    ):
        with pytest.raises(ValueError, match="the real failure"):
            with AdapterMixin.adapter_scope(mock_backend, adapter):
                raise ValueError("the real failure")

    weights.deactivate.assert_called_once()


def test_adapter_scope_is_not_atomic_across_concurrent_calls():
    """Known limitation, not a guarantee: two concurrent `adapter_scope()`
    calls on one backend can interleave.

    `_adapter_activation_lock()` is held only inside each of
    `activate()`/`deactivate()`'s own verb calls, not across the `with` body
    in between — so a second thread's full activate-body-deactivate cycle can
    run while the first thread's body is still executing, leaving the first
    thread's body observing a different adapter (or none) active.

    Widening the lock to span the whole scope was tried and reverted: it
    deadlocks the real async generation path (see the docstring note on
    `adapter_scope`). This test pins today's actual (non-atomic) behaviour so
    it doesn't get silently "fixed" back to interleaving by an unrelated
    change, or silently broken worse. #1465 owns making this atomic, together
    with the threading model for real generation.
    """
    import threading

    class _FakeBackend:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.active: str | None = None

        def _adapter_activation_lock(self):
            return self._lock

        def activate_peft_adapter(self, name: str) -> None:
            self.active = name

        def deactivate_peft_adapter(self, name: str) -> None:
            self.active = None

    def _make(backend: _FakeBackend, name: str):
        binding = LocalFileBinding(name=name, adapter_type=AdapterType.LORA)
        binding.backend = backend  # type: ignore[assignment]
        binding._loaded = (
            True  # bypass prepare(); this test is about activate/deactivate
        )
        identity = Identity(name=name, adapter_type="lora")
        return Adapter(identity=identity, io_contract=_Contract(), weights=binding)

    backend = _FakeBackend()
    a1 = _make(backend, "adapter_one")
    a2 = _make(backend, "adapter_two")

    observed_inside_a1_body: list[str | None] = []
    a1_activated = threading.Event()
    a2_done = threading.Event()

    def thread1() -> None:
        with AdapterMixin.adapter_scope(backend, a1):
            a1_activated.set()
            a2_done.wait(timeout=2)
            observed_inside_a1_body.append(backend.active)

    def thread2() -> None:
        a1_activated.wait(timeout=2)
        with AdapterMixin.adapter_scope(backend, a2):
            pass
        a2_done.set()

    t1 = threading.Thread(target=thread1)
    t2 = threading.Thread(target=thread2)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    # If adapter_scope were atomic, this would still be "adapter_one_lora".
    # It isn't: thread2's full cycle ran to completion (and deactivated)
    # while thread1's body was still executing.
    assert observed_inside_a1_body == [None]


def test_adapter_scope_noop_when_adapter_is_none():
    mock_backend = MagicMock(spec=AdapterMixin)

    entered = False
    with AdapterMixin.adapter_scope(mock_backend, None):
        entered = True

    assert entered


if __name__ == "__main__":
    pytest.main([__file__])
