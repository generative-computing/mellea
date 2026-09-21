# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the adapter resolution chain against a real HF repo.

Only `io.yaml` files are fetched (`obtain_io_yaml` scopes the download), so these
need no model, GPU, or adapter weights — just Hub read access. They are gated on a
Hugging Face token; the repo is private, so they run locally for a logged-in
developer and skip in CI until the layout lands in a repo CI can read.
"""

import functools
import logging
from unittest.mock import MagicMock

import pytest

from mellea.backends.adapters._core import LocalFileBinding, PromptBinding
from mellea.backends.adapters.adapter import _register_available_composed_adapter
from mellea.backends.adapters.catalog import AdapterType, IntrinsicsCatalogEntry
from mellea.formatters.granite.intrinsics.util import (
    obtain_io_yaml,
    resolve_prompt_io_yaml,
)
from test.predicates import require_hf_token

pytestmark = [pytest.mark.e2e, pytest.mark.huggingface, require_hf_token()]

# TODO: temporary repo; replace with a public repo once it carries this layout.
_TEST_REPO = "ajbozarth/granitelib-rag-r1.0"
_TEST_SHA = "8bad850fc9855fd655cfd34deb44ddef1597255e"  # main @ 2026-09-21


# --- Private-repo access guard (temporary; remove with the TODO above) -----------
# The fixture repo is private, so these tests xfail rather than hard-fail when it is
# unreadable (CI's public-only token, or another developer's), and run only with
# access. Goes inert once the repo is public.


@functools.cache
def _test_repo_readable() -> bool:
    """Whether the current HF token can read the test repo (checked once)."""
    try:
        from huggingface_hub import HfApi

        HfApi().repo_info(_TEST_REPO, revision=_TEST_SHA)
    except Exception:
        return False
    return True


@pytest.fixture(autouse=True)
def _xfail_if_test_repo_unreadable():
    """xfail the test when the test repo is unreadable with the current token."""
    if not _test_repo_readable():
        pytest.xfail(f"{_TEST_REPO} not readable with the current token (private)")


def _test_entry(capability: str = "answerability") -> IntrinsicsCatalogEntry:
    """A catalog entry for `capability`."""
    return IntrinsicsCatalogEntry(
        name=capability,
        repo_id=_TEST_REPO,
        revision=_TEST_SHA,
        adapter_types=(AdapterType.ALORA, AdapterType.LORA),
    )


# --- Low-level: obtain_io_yaml / resolve_prompt_io_yaml against the test repo -------


def test_trained_io_yaml_resolves():
    """A trained aLoRA adapter's io.yaml resolves (answerability/granite-4.1-3b)."""
    path = obtain_io_yaml(
        "answerability",
        "granite-4.1-3b",
        _TEST_REPO,
        revision=_TEST_SHA,
        adapter_type="alora",
    )
    assert path.exists()
    assert path.name == "io.yaml"


def test_lora_only_io_yaml_resolves():
    """citations is LoRA-only on disk: the aLoRA probe misses, the LoRA resolves.

    The absent type surfaces as the not-found `ValueError` the walk advances on;
    the present type resolves normally.
    """
    with pytest.raises(ValueError):
        obtain_io_yaml(
            "citations",
            "granite-4.1-3b",
            _TEST_REPO,
            revision=_TEST_SHA,
            adapter_type="alora",
        )
    path = obtain_io_yaml(
        "citations",
        "granite-4.1-3b",
        _TEST_REPO,
        revision=_TEST_SHA,
        adapter_type="lora",
    )
    assert path.exists()
    assert path.name == "io.yaml"


def test_model_specific_prompt_resolves():
    """granite-4.2-8b has no trained adapter, but its model-specific prompt resolves."""
    path, used_any = resolve_prompt_io_yaml(
        "answerability", "granite-4.2-8b", _TEST_REPO, revision=_TEST_SHA
    )
    assert path.exists()
    assert used_any is False


def test_any_prompt_fallback_resolves():
    """granite-4.2-3b has nothing of its own, so resolution falls back to `any`."""
    path, used_any = resolve_prompt_io_yaml(
        "answerability", "granite-4.2-3b", _TEST_REPO, revision=_TEST_SHA
    )
    assert path.exists()
    assert used_any is True


def test_bad_revision_aborts_not_a_valueerror():
    """A bad revision raises a Hub error, not the not-found ValueError.

    The not-found `ValueError` is the fallthrough signal; a genuine error (here a
    nonexistent revision) must be distinguishable so resolution aborts instead of
    silently advancing down the chain.
    """
    with pytest.raises(Exception) as exc_info:
        obtain_io_yaml(
            "answerability",
            "granite-4.2-8b",
            _TEST_REPO,
            revision="0" * 40,  # nonexistent commit sha
            adapter_type="prompt",
        )
    assert not isinstance(exc_info.value, ValueError)


# --- Higher-level: the resolution walk against the test repo (mock backend) --------
#
# A MagicMock backend means the walk really probes the test repo but its add_adapter
# is a no-op, so no weights download or model load happens — we assert only what
# the walk *resolves to* for each rung.


def test_walk_resolves_trained_adapter():
    """granite-4.1-3b resolves to a LocalFileBinding.

    Both aLoRA and LoRA exist for this model; aLoRA is probed first and wins.
    """
    backend = MagicMock()
    _register_available_composed_adapter(
        backend, "answerability", "granite-4.1-3b", _test_entry()
    )
    adapter = backend.add_adapter.call_args.args[0]
    assert isinstance(adapter.weights, LocalFileBinding)
    assert adapter.weights.adapter_type is AdapterType.ALORA


def test_walk_resolves_lora_only_adapter():
    """citations/granite-4.1-3b resolves to a LoRA LocalFileBinding.

    Only `lora/` exists on disk, so the aLoRA probe misses and the walk advances
    to LoRA.
    """
    backend = MagicMock()
    _register_available_composed_adapter(
        backend, "citations", "granite-4.1-3b", _test_entry("citations")
    )
    adapter = backend.add_adapter.call_args.args[0]
    assert isinstance(adapter.weights, LocalFileBinding)
    assert adapter.weights.adapter_type is AdapterType.LORA


def test_walk_resolves_model_specific_prompt():
    """granite-4.2-8b resolves to a PromptBinding."""
    backend = MagicMock()
    _register_available_composed_adapter(
        backend, "answerability", "granite-4.2-8b", _test_entry()
    )
    backend.add_adapter.assert_called_once()
    adapter = backend.add_adapter.call_args.args[0]
    assert isinstance(adapter.weights, PromptBinding)
    assert adapter.weights.target_model_name == "granite-4.2-8b"
    assert isinstance(backend.add_adapter.call_args.kwargs["config"], dict)


def test_walk_resolves_any_prompt_with_warning(caplog):
    """granite-4.2-3b resolves to the `any` prompt and warns."""
    backend = MagicMock()
    with caplog.at_level(logging.WARNING, logger="mellea"):
        _register_available_composed_adapter(
            backend, "answerability", "granite-4.2-3b", _test_entry()
        )
    adapter = backend.add_adapter.call_args.args[0]
    assert isinstance(adapter.weights, PromptBinding)
    assert adapter.weights.target_model_name == "any"
    assert any("model-agnostic 'any'" in r.message for r in caplog.records)
