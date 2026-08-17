# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LocalFileBinding (Epic #929 Phase 2, issue #1141).

Uses a fake AdapterMixin-conforming backend double throughout — no real HF
model or network access.
"""

import threading
from collections.abc import Coroutine
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mellea.backends.adapters._core import LocalFileBinding
from mellea.backends.adapters.catalog import AdapterType, fetch_intrinsic_metadata


def _fake_backend():
    """A minimal AdapterMixin-conforming double."""
    backend = MagicMock()
    backend.add_adapter.side_effect = lambda binding: setattr(
        binding, "backend", backend
    )
    return backend


def test_construction_defaults():
    binding = LocalFileBinding()
    assert binding.name == ""
    assert binding.adapter_type is AdapterType.LORA
    assert binding.repo_id == ""
    # `None`, not "main": a default-constructed binding must not silently opt into
    # tracking-latest. `None` defers to the catalogue's pinned revision.
    assert binding.revision is None
    assert binding.backend is None
    assert binding.path is None


def test_resolved_revision_falls_back_to_catalogue_pin():
    pinned = fetch_intrinsic_metadata("answerability").revision
    binding = LocalFileBinding(name="answerability")
    assert binding.revision is None
    assert binding.resolved_revision() == pinned
    assert binding.resolved_revision() != "main"


def test_resolved_revision_honours_explicit_main_override():
    binding = LocalFileBinding(name="answerability", revision="main")
    assert binding.resolved_revision() == "main"


def test_resolved_revision_unknown_name_raises():
    binding = LocalFileBinding(name="not-a-real-adapter-function")
    with pytest.raises(ValueError, match="Unknown intrinsic name"):
        binding.resolved_revision()


def test_prepare_rejects_unconfigured_binding():
    backend = _fake_backend()
    binding = LocalFileBinding()
    binding.bind_backend(backend)
    with pytest.raises(RuntimeError, match="requires a non-empty name"):
        binding.prepare()


def test_qualified_name():
    binding = LocalFileBinding(name="answerability", adapter_type=AdapterType.ALORA)
    assert binding.qualified_name == "answerability_alora"


def test_from_catalog_uses_pinned_metadata():
    metadata = fetch_intrinsic_metadata("answerability")

    binding = LocalFileBinding.from_catalog("answerability")

    assert binding.name == "answerability"
    assert binding.repo_id == metadata.repo_id
    assert binding.revision == metadata.revision
    assert binding.revision != "main"
    assert binding.adapter_type == metadata.adapter_types[0]


def test_from_catalog_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown intrinsic name"):
        LocalFileBinding.from_catalog("not-a-real-adapter-function")


def test_prepare_without_bind_backend_raises():
    binding = LocalFileBinding(name="answerability")
    with pytest.raises(RuntimeError, match="bind_backend"):
        binding.prepare()


def test_prepare_registers_and_loads_on_staged_backend():
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)

    binding.prepare()

    backend.add_adapter.assert_called_once_with(binding)
    backend.load_peft_adapter.assert_called_once_with(binding.qualified_name)
    assert binding.backend is backend


def test_prepare_is_idempotent():
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)

    binding.prepare()
    binding.prepare()

    backend.add_adapter.assert_called_once()
    backend.load_peft_adapter.assert_called_once()


def test_prepare_ignores_phase_hook_dispatch_failure():
    """A prepare hook failure must not make successfully loaded weights unusable."""
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)

    with (
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.plugins.hooks.adapter_function.AdapterFunctionPhaseCompletePayload",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        patch(
            "mellea.backends.adapters._core.invoke_hook",
            side_effect=RuntimeError("plugin dispatch blew up"),
        ),
    ):
        binding.prepare()

    assert binding._loaded
    binding.activate()
    backend.load_peft_adapter.assert_called_once_with(binding.qualified_name)
    backend.activate_peft_adapter.assert_called_once_with(binding.qualified_name)


def test_prepare_retries_only_the_load_after_a_load_failure():
    """A failed load must be retryable without re-registering.

    Regression guard: `add_adapter` sets `.backend` (registration) before
    `prepare()` calls `load_peft_adapter` (the load). If the load raised,
    `.backend` was already non-None, so the old idempotency guard
    (`if self.backend is not None: return`) made every retry a silent no-op —
    the caller got no error and no adapter, forever. The fix tracks the load
    separately from registration so a retry redoes only the failed step.
    """
    backend = _fake_backend()
    backend.load_peft_adapter.side_effect = [
        RuntimeError("transient load failure"),
        None,
    ]
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)

    with pytest.raises(RuntimeError, match="transient load failure"):
        binding.prepare()

    # Registration succeeded (that's why .backend is set); the load did not.
    # A binding in this state must not look "already prepared".
    assert binding.backend is backend
    with pytest.raises(RuntimeError, match="prepare"):
        binding.activate()

    binding.prepare()  # retry: must not re-register, must retry the load

    backend.add_adapter.assert_called_once()
    assert backend.load_peft_adapter.call_count == 2
    binding.activate()
    backend.activate_peft_adapter.assert_called_once_with(binding.qualified_name)


def test_bind_backend_after_release_raises():
    """release() is terminal: bind_backend() must not silently revive the binding."""
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()
    binding.release()

    other_backend = _fake_backend()
    with pytest.raises(RuntimeError, match="release"):
        binding.bind_backend(other_backend)


def test_prepare_after_release_raises():
    """release() is terminal: prepare() must not silently revive the binding."""
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()
    binding.release()

    # Bypass bind_backend()'s own guard to confirm prepare() enforces this too.
    binding._staged_backend = _fake_backend()
    with pytest.raises(RuntimeError, match="release"):
        binding.prepare()


def test_activate_without_prepare_raises():
    binding = LocalFileBinding(name="answerability")
    with pytest.raises(RuntimeError, match="prepare"):
        binding.activate()


def test_deactivate_without_prepare_raises():
    binding = LocalFileBinding(name="answerability")
    with pytest.raises(RuntimeError, match="prepare"):
        binding.deactivate()


def test_activate_delegates_to_backend_verb():
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()

    binding.activate()

    backend.activate_peft_adapter.assert_called_once_with(binding.qualified_name)


def test_deactivate_delegates_to_backend_verb():
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()

    binding.deactivate()

    backend.deactivate_peft_adapter.assert_called_once_with(binding.qualified_name)


def test_activate_holds_the_backends_activation_lock():
    """`activate()` must hold whatever lock `_adapter_activation_lock()` returns.

    `activate_peft_adapter`/`deactivate_peft_adapter` document "must be called
    while holding `_generation_lock`" as a precondition on the backend side;
    `_adapter_activation_lock()` is the only thing satisfying that precondition
    on this path (`adapter_scope` holds no lock of its own). A real
    `threading.Lock` proves it's actually held during the call, not just
    entered-and-exited around a no-op.
    """
    backend = _fake_backend()
    lock = threading.Lock()
    backend._adapter_activation_lock.return_value = lock
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()

    observed_locked = {}
    backend.activate_peft_adapter.side_effect = lambda _name: (
        observed_locked.setdefault("during_call", lock.locked())
    )

    binding.activate()

    assert observed_locked["during_call"] is True
    assert not lock.locked()


def test_deactivate_holds_the_backends_activation_lock():
    backend = _fake_backend()
    lock = threading.Lock()
    backend._adapter_activation_lock.return_value = lock
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()

    observed_locked = {}
    backend.deactivate_peft_adapter.side_effect = lambda _name: (
        observed_locked.setdefault("during_call", lock.locked())
    )

    binding.deactivate()

    assert observed_locked["during_call"] is True
    assert not lock.locked()


def test_release_without_prepare_is_noop():
    binding = LocalFileBinding(name="answerability")
    binding.release()  # must not raise


def test_release_after_bind_before_prepare_clears_staged_backend():
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)

    binding.release()

    assert binding._staged_backend is None
    assert binding._released
    backend.unload_peft_adapter.assert_not_called()


def test_release_unloads_and_clears_state():
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()

    binding.release()

    backend.unload_peft_adapter.assert_called_once_with(binding.qualified_name)
    assert binding.backend is None
    assert binding.path is None
    assert binding._staged_backend is None


def test_release_retries_after_unload_failure():
    backend = _fake_backend()
    backend.unload_peft_adapter.side_effect = [
        RuntimeError("transient unload failure"),
        None,
    ]
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()
    binding.path = "/fake/adapter"

    with pytest.raises(RuntimeError, match="transient unload failure"):
        binding.release()

    assert not binding._released
    assert binding.backend is backend
    assert binding.path == "/fake/adapter"
    assert binding._staged_backend is backend
    assert binding._loaded
    binding.activate()

    binding.release()

    assert backend.unload_peft_adapter.call_count == 2
    assert binding._released
    assert binding.backend is None
    assert binding.path is None
    assert binding._staged_backend is None
    assert not binding._loaded


def test_release_is_idempotent():
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()

    binding.release()
    binding.release()

    backend.unload_peft_adapter.assert_called_once()


def test_prepare_fires_phase_complete_metric_when_plugins_present():
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)

    with (
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch("mellea.backends.adapters._core._run_async_in_thread") as mock_run,
    ):
        binding.prepare()

    mock_run.assert_called_once()
    hook_coro = mock_run.call_args.args[0]
    assert isinstance(hook_coro, Coroutine)
    hook_coro.close()


def test_release_does_not_fire_phase_complete_metric():
    # "release" is not a valid AdapterFunctionPhaseCompletePayload.phase value.
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    backend = _fake_backend()
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()

    with (
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch("mellea.backends.adapters._core._run_async_in_thread") as mock_run,
    ):
        binding.release()

    mock_run.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
