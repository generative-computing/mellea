# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the narrowed AdapterMixin verb contract (Epic #929 Phase 2, issue #1140).

Verifies that:
  - the four reality-specific verbs (`load_peft_adapter`, `unload_peft_adapter`,
    `render_controls`, `set_request_adapter`) raise `NotImplementedError` by
    default on the mixin
  - each concrete backend overrides only the verb(s) matching its own adapter
    reality, leaving the others on the default (raising) implementation
"""

from unittest.mock import MagicMock

import pytest

from mellea.backends.adapters.adapter import AdapterMixin
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.openai import OpenAIBackend

_REALITY_SPECIFIC_VERBS = (
    "load_peft_adapter",
    "unload_peft_adapter",
    "render_controls",
    "set_request_adapter",
)


@pytest.mark.parametrize("verb", _REALITY_SPECIFIC_VERBS)
def test_default_reality_specific_verb_raises_not_implemented(verb):
    """Each reality-specific verb raises NotImplementedError by default on the mixin."""
    mock_backend = MagicMock(spec=AdapterMixin)
    method = getattr(AdapterMixin, verb)
    args = ("some_adapter", True) if verb == "render_controls" else ("some_adapter",)

    with pytest.raises(NotImplementedError):
        method(mock_backend, *args)


def test_hf_backend_overrides_only_peft_verbs():
    """LocalHFBackend (LocalFile/PEFT reality) overrides load/unload_peft_adapter only."""
    assert "load_peft_adapter" in vars(LocalHFBackend)
    assert "unload_peft_adapter" in vars(LocalHFBackend)
    assert "render_controls" not in vars(LocalHFBackend)
    assert "set_request_adapter" not in vars(LocalHFBackend)


def test_openai_backend_overrides_only_render_controls():
    """OpenAIBackend (Embedded/Granite Switch reality) overrides render_controls only."""
    assert "render_controls" in vars(OpenAIBackend)
    assert "load_peft_adapter" not in vars(OpenAIBackend)
    assert "unload_peft_adapter" not in vars(OpenAIBackend)
    assert "set_request_adapter" not in vars(OpenAIBackend)


def test_no_backend_implements_server_mediated_reality():
    """set_request_adapter has no concrete implementation anywhere yet."""
    assert "set_request_adapter" not in vars(LocalHFBackend)
    assert "set_request_adapter" not in vars(OpenAIBackend)
