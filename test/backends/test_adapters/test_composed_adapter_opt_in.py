# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`_supports_composed_adapters` opt-in on the in-tree backends (issue #1824).

`AdapterMixin` defaults the flag to `False`, and `resolve_adapter` refuses to
register an adapter function on a backend that has not opted in. Losing the
opt-in on an in-tree backend therefore breaks every non-embedded adapter function
call on it.
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from mellea.backends.adapters.adapter import AdapterMixin


@pytest.mark.parametrize(
    ("module_name", "class_name", "extra"),
    [
        ("mellea.backends.huggingface", "LocalHFBackend", "torch"),
        ("mellea.backends.openai", "OpenAIBackend", "openai"),
        ("mellea.backends.ollama", "OllamaModelBackend", "ollama"),
    ],
)
def test_in_tree_backend_opts_into_composed_adapters(
    module_name: str, class_name: str, extra: str
):
    """Each in-tree adapter backend must set `_supports_composed_adapters = True`."""
    pytest.importorskip(extra)
    backend_cls = getattr(importlib.import_module(module_name), class_name)
    assert issubclass(backend_cls, AdapterMixin)
    assert backend_cls._supports_composed_adapters is True


def test_adapter_mixin_defaults_composed_adapters_to_false():
    """The opt-in defaults to `False`, so a backend must set it explicitly."""
    assert AdapterMixin._supports_composed_adapters is False


def test_resolve_adapter_raises_without_composed_adapter_support():
    """`resolve_adapter` raises `ValueError` on a backend that has not opted in."""
    backend = MagicMock(spec=AdapterMixin)
    backend._supports_composed_adapters = False
    backend._uses_embedded_adapters = False
    backend.base_model_name = "granite-4.1-3b"
    backend._find_adapter.return_value = None

    with (
        patch("mellea.backends.adapters.adapter.fetch_intrinsic_metadata"),
        pytest.raises(ValueError, match="does not support composed adapters"),
    ):
        AdapterMixin.resolve_adapter(backend, "answerability")
    backend.add_adapter.assert_not_called()
