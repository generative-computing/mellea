# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real e2e test: the intrinsic `adapter_scope` path against a real PEFT model.

PR #1555 (#1465) routed `LocalHFBackend`'s LoRA/aLoRA intrinsic generation
through `AdapterMixin.adapter_scope()`, via `_generate_intrinsic_with_adapter_scope`
and `_IntrinsicPeftBinding`. That plumbing is covered by `MagicMock`-based unit
tests (`test/backends/test_huggingface_unit.py`) that pin verb order, lock
behaviour, and hook payload shape precisely, but never against a real
`PeftModel` or a real downloaded adapter. `test_local_file_e2e.py` proves
`adapter_scope()` against a real model, but drives `LocalFileBinding` directly,
not the intrinsic path.

This test drives one real intrinsic call (`core.check_certainty`) through the
public API end to end against a real Granite model and the real
`ibm-granite/granitelib-core-r1.0` "uncertainty" adapter, and asserts only on
the `ADAPTER_FUNCTION_*` hook payloads and real model adapter state — never on
the generated certainty score itself. See test/README.md's e2e rules and #1291
(the flakiness class that score-content assertions produced in GPU intrinsic
tests). This does not make the test immune to model output: `check_certainty`
still requires the adapter's raw output to parse as JSON containing a
`certainty` key (`core.py`), so a real parse failure surfaces as an
exception from the `core.check_certainty(...)` call itself, before any hook
assertion runs — `hf_skip()` only converts Hub network errors to skips, not
this. `call_intrinsic` pins `temperature=0.0` for exactly this reason, which
keeps this residual risk low but not zero.

Also proves, against the real PEFT model, that a second call to the same
intrinsic succeeds: `load_peft_adapter`'s tolerance of PEFT's "Adapter with
name ... already exists" `ValueError` on a repeat load is a path the mocks in
`test_huggingface_unit.py` cannot exercise, since nothing there ever loads
twice into a real `PeftModel`.
"""

import os
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

from test.predicates import require_gpu

pytestmark = [
    pytest.mark.huggingface,
    pytest.mark.e2e,
    pytest.mark.slow,
    # 12GB, not test_local_file_e2e.py's 20GB: matches the existing bound for
    # this same model (granite-4.1-3b) in test_core.py, not drift.
    require_gpu(min_vram_gb=12),
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping HuggingFace e2e tests in CI",
    ),
]

from mellea.backends import model_ids
from mellea.backends.adapters.catalog import fetch_intrinsic_metadata
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext
from test.backends.test_adapters._hook_capture import (
    capture_adapter_hooks,
    invocation_payloads,
    phase_payloads,
)
from test.conftest import cleanup_gpu_backend, hf_skip

_UNCERTAINTY_REVISION = fetch_intrinsic_metadata("uncertainty").revision


@pytest.fixture
def backend():
    with hf_skip():
        backend_ = LocalHFBackend(model_id=model_ids.IBM_GRANITE_4_1_3B)
    yield backend_
    cleanup_gpu_backend(backend_, backend_name="intrinsic_adapter_scope_e2e")


@pytest.fixture
def chat_context(backend):
    """A real user+assistant turn, satisfying `_assert_context_forwards_history`."""
    with hf_skip():
        _, ctx = mfuncs.chat("What is 2 + 2?", ChatContext(), backend, model_options={})
    return ctx


def _assert_single_success_invocation(mock_invoke: MagicMock) -> None:
    """Asserts exactly one activate/deactivate phase pair, and one invocation
    with the expected outcome, name, adapter_type, binding_type, and revision."""
    phases = [p.phase for p in phase_payloads(mock_invoke)]
    assert phases == ["activate", "deactivate"]

    invocations = invocation_payloads(mock_invoke)
    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.outcome == "success"
    assert invocation.name == "uncertainty"
    # resolve_adapter's lazy registration always registers AdapterType.LORA
    # for this (non-embedded) path, regardless of which types the catalog
    # lists as available (adapter.py's resolve_adapter, "pre-Phase-1
    # default" comment) — "lora" is the real value on this call path, not
    # "unknown" (the payload field's default if this were ever omitted).
    assert invocation.adapter_type == "lora"
    assert invocation.binding_type == "local_file"
    assert invocation.revision == _UNCERTAINTY_REVISION


def test_check_certainty_adapter_scope_fires_hooks_and_activates_real_adapter(
    backend, chat_context
):
    with hf_skip(), capture_adapter_hooks() as mock_invoke:
        core.check_certainty(chat_context, backend)

    _assert_single_success_invocation(mock_invoke)

    # Real PEFT model, cleanly deactivated after the scope exits.
    assert backend._model.active_adapters() == []  # type: ignore[union-attr]


def test_check_certainty_adapter_scope_repeat_call_is_idempotent(backend, chat_context):
    """A second call proves `load_peft_adapter`'s already-loaded tolerance
    against a real model, not just the mocked `ValueError` string match."""
    with hf_skip(), capture_adapter_hooks() as first_invoke:
        core.check_certainty(chat_context, backend)
    _assert_single_success_invocation(first_invoke)

    with hf_skip(), capture_adapter_hooks() as second_invoke:
        core.check_certainty(chat_context, backend)
    _assert_single_success_invocation(second_invoke)

    assert backend._model.active_adapters() == []  # type: ignore[union-attr]


if __name__ == "__main__":
    pytest.main([__file__])
