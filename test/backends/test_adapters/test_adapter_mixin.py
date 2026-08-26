# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the narrowed AdapterMixin verb contract (Epic #929 Phase 2, issue #1140).

Verifies that:
  - the reality-specific verbs (`load_peft_adapter`, `unload_peft_adapter`,
    `remove_adapter`, `activate_peft_adapter`, `deactivate_peft_adapter`)
    raise `NotImplementedError` by default on the mixin
  - each concrete backend overrides only the verb(s) matching its own adapter
    reality, leaving the others on the default (raising) implementation

Embedded/Granite Switch activation is not a mixin verb — it lives on
`EmbeddedBinding.apply_activation` (issue #1142); see test_embedded_binding.py.
"""

from unittest.mock import MagicMock

import pytest

from mellea.backends.adapters.adapter import AdapterMixin
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.openai import OpenAIBackend

_REALITY_SPECIFIC_VERBS = (
    "load_peft_adapter",
    "unload_peft_adapter",
    "remove_adapter",
    "activate_peft_adapter",
    "deactivate_peft_adapter",
)


@pytest.mark.parametrize("verb", _REALITY_SPECIFIC_VERBS)
def test_default_reality_specific_verb_raises_not_implemented(verb):
    """Each reality-specific verb raises NotImplementedError by default on the mixin."""
    mock_backend = MagicMock(spec=AdapterMixin)
    method = getattr(AdapterMixin, verb)

    with pytest.raises(NotImplementedError):
        method(mock_backend, "some_adapter")


def test_hf_backend_overrides_only_peft_verbs():
    """LocalHFBackend (LocalFile/PEFT reality) overrides the PEFT verbs only."""
    assert "load_peft_adapter" in vars(LocalHFBackend)
    assert "unload_peft_adapter" in vars(LocalHFBackend)
    assert "remove_adapter" in vars(LocalHFBackend)
    assert "activate_peft_adapter" in vars(LocalHFBackend)
    assert "deactivate_peft_adapter" in vars(LocalHFBackend)


def test_openai_backend_overrides_no_reality_specific_verbs():
    """OpenAIBackend (Embedded/Granite Switch reality) activates through the
    binding, not a mixin verb, so it overrides none of the PEFT verbs."""
    assert "load_peft_adapter" not in vars(OpenAIBackend)
    assert "unload_peft_adapter" not in vars(OpenAIBackend)
    assert "remove_adapter" not in vars(OpenAIBackend)
    assert "activate_peft_adapter" not in vars(OpenAIBackend)
    assert "deactivate_peft_adapter" not in vars(OpenAIBackend)


def test_default_adapter_activation_lock_is_a_noop():
    """The mixin default is a no-op context manager, not a real lock.

    Backends whose activation verbs mutate shared, non-thread-safe state
    override this; a backend that doesn't (nothing to protect) gets a
    `nullcontext` for free rather than having to implement one.
    """
    mock_backend = MagicMock(spec=AdapterMixin)

    with AdapterMixin._adapter_activation_lock(mock_backend):
        pass


def test_hf_backend_overrides_adapter_activation_lock():
    """LocalHFBackend overrides the lock (it has shared PEFT model state to protect)."""
    assert "_adapter_activation_lock" in vars(LocalHFBackend)


def test_openai_backend_does_not_override_adapter_activation_lock():
    """OpenAIBackend has no PEFT model state, so it keeps the mixin's no-op default."""
    assert "_adapter_activation_lock" not in vars(OpenAIBackend)
