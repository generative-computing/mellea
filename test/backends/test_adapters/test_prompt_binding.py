# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the weightless PromptBinding and the PROMPT adapter type."""

import logging

import pytest

from mellea.backends.adapters._core import Identity, PromptBinding
from mellea.backends.adapters.catalog import AdapterType


def test_adapter_type_prompt_value():
    assert AdapterType.PROMPT.value == "prompt"


def test_identity_accepts_prompt():
    identity = Identity(
        name="answerability", adapter_type="prompt", capability="answerability"
    )
    assert identity.adapter_type == "prompt"


def test_identity_rejects_unknown_type():
    with pytest.raises(ValueError):
        Identity(name="answerability", adapter_type="bogus")  # type: ignore[arg-type]


def test_prompt_binding_qualified_name():
    binding = PromptBinding(name="answerability")
    assert binding.qualified_name == "answerability_prompt"


def test_prompt_binding_lifecycle_verbs_are_noops():
    binding = PromptBinding(name="answerability")
    # Genuine no-ops: return nothing and never raise.
    assert binding.prepare() is None
    assert binding.deactivate() is None
    assert binding.release() is None


def test_prompt_binding_activate_logs_at_info(caplog):
    binding = PromptBinding(name="answerability", target_model_name="any")
    with caplog.at_level(logging.INFO, logger="mellea"):
        binding.activate()
    assert any(
        "Prompt fallback activated" in r.message and "answerability" in r.message
        for r in caplog.records
    )


def test_prompt_binding_resolved_revision_prefers_explicit():
    binding = PromptBinding(name="answerability", revision="abc123")
    assert binding.resolved_revision() == "abc123"


def test_prompt_binding_resolved_revision_falls_back_to_catalog():
    # None revision resolves via the catalogue's pinned revision for the name.
    binding = PromptBinding(name="answerability", revision=None)
    assert isinstance(binding.resolved_revision(), str)
    assert binding.resolved_revision() != ""
