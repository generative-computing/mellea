# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for LocalHFBackend's <think>...</think> tag splitting.

Unlike test_huggingface_thinking.py (pure unit tests against synthetic strings),
these tests run real generation against granite-4.2-3b to verify the raw text
produced by `model.generate()` + `tokenizer.decode(skip_special_tokens=True)`
actually has the shape _split_think_tags assumes. This matters because <think>
and </think> are registered as *non-special* added tokens in granite-4.2's
tokenizer (special: false), so skip_special_tokens=True does not strip them —
but that is a property of this specific model/tokenizer, not something a
synthetic-string test can verify.
"""

import os

import pytest

from test.predicates import require_gpu

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

pytestmark = [
    pytest.mark.huggingface,
    pytest.mark.e2e,
    pytest.mark.qualitative,
    require_gpu(min_vram_gb=8),
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping HuggingFace thinking e2e tests in CI - qualitative test",
    ),
]

import mellea.backends.model_ids as model_ids
from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.context import ChatContext
from test.conftest import hf_skip


@pytest.fixture(scope="module")
def backend():
    """Shared granite-4.2-3b HuggingFace backend for all tests in this module."""
    with hf_skip():
        backend = LocalHFBackend(
            model_id=model_ids.IBM_GRANITE_4_2_3B, cache=SimpleLRUCache(5)
        )
    yield backend

    from test.conftest import cleanup_gpu_backend

    cleanup_gpu_backend(backend, "huggingface-thinking")


@pytest.fixture(scope="function")
def session(backend):
    """Fresh HuggingFace session for each test."""
    session = MelleaSession(backend, ctx=ChatContext())
    yield session
    session.reset()


def test_thinking_enabled_populates_mot_thinking(session):
    """ModelOption.THINKING=True: mot.thinking holds the real reasoning trace,
    mot.value is the clean answer with no leftover </think> tag, and the answer
    itself still contains the expected content (not just an empty string that
    would vacuously satisfy the tag-absence checks alone)."""
    output = session.instruct(
        "What is 2 + 2? Answer with just the number.",
        model_options={ModelOption.THINKING: True, ModelOption.MAX_NEW_TOKENS: 400},
    )
    assert output.thinking, (
        f"Expected a non-empty reasoning trace, got: {output.thinking!r}"
    )
    # granite-4.2's chat template bakes the opening <think> into the generation
    # prompt itself (see module docstring / _split_think_tags), so the model's own
    # output never contains it — checking for its absence here would be vacuous.
    # </think> is the tag that actually appears in raw output and must be split out.
    assert "</think>" not in output.value
    assert "4" in output.value


def test_thinking_disabled_leaves_mot_thinking_none(session):
    """ModelOption.THINKING=False: no think block is generated, so mot.thinking
    is falsy (None or empty string — both mean "no reasoning trace captured")."""
    output = session.instruct(
        "What is 2 + 2? Answer with just the number.",
        model_options={ModelOption.THINKING: False, ModelOption.MAX_NEW_TOKENS: 100},
    )
    assert not output.thinking, f"Expected no reasoning trace, got: {output.thinking!r}"
    assert "</think>" not in output.value
    assert "4" in output.value
