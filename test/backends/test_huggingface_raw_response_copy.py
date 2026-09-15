# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for copy/deepcopy semantics of ModelOutputThunks that hold an HF
GenerateDecoderOnlyOutput in raw.response — the raw batch path specific case."""

import copy
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)

from transformers.generation.utils import GenerateDecoderOnlyOutput

from mellea.core import ModelOutputThunk


def _make_mot_with_hf_raw_response() -> tuple[ModelOutputThunk, Any]:
    """Build a MOT whose raw.response mirrors what the raw batch path stores.

    The batch path slices one row from the full-batch sequences tensor and
    immediately calls .detach().clone(), so raw.response.sequences is always
    an owning, contiguous tensor — never a view.
    """
    full_batch = torch.arange(6, dtype=torch.long).reshape(2, 3)
    sequences = full_batch[0:1, :].detach().clone()
    hf_out = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=None,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )
    mot = ModelOutputThunk(value="hello")
    mot.raw.response = hf_out
    return mot, sequences


def test_shallow_copy_raw_response_is_same_object():
    """copy.copy(mot).raw.response is the same object as mot.raw.response."""
    mot, _ = _make_mot_with_hf_raw_response()
    copied = copy.copy(mot)
    assert copied.raw.response is mot.raw.response, (
        "shallow copy must keep raw.response as the same object"
    )


def test_shallow_copy_raw_response_sequences_shares_storage():
    """Shallow-copied MOT shares the same sequences tensor as the original."""
    mot, original_sequences = _make_mot_with_hf_raw_response()
    copied = copy.copy(mot)
    assert (
        copied.raw.response.sequences.untyped_storage().data_ptr()
        == original_sequences.untyped_storage().data_ptr()
    ), "shallow copy: raw.response.sequences must share storage with the original MOT"


def test_deepcopy_raw_response_is_distinct_object():
    """copy.deepcopy(mot).raw.response is a distinct object from mot.raw.response."""
    mot, _ = _make_mot_with_hf_raw_response()
    deep = copy.deepcopy(mot)
    assert deep.raw.response is not mot.raw.response, (
        "deepcopy must produce a new raw.response object"
    )


def test_deepcopy_raw_response_sequences_does_not_share_storage():
    """Deepcopy breaks tensor storage sharing for raw.response.sequences."""
    mot, original_sequences = _make_mot_with_hf_raw_response()
    deep = copy.deepcopy(mot)
    assert (
        deep.raw.response.sequences.untyped_storage().data_ptr()
        != original_sequences.untyped_storage().data_ptr()
    ), "deepcopy: raw.response.sequences must NOT share storage with the original"


def test_deepcopy_raw_response_sequences_preserves_values():
    """Deepcopy preserves tensor values in raw.response.sequences despite storage isolation."""
    mot, _ = _make_mot_with_hf_raw_response()
    original_values = mot.raw.response.sequences.clone()
    deep = copy.deepcopy(mot)
    assert torch.equal(deep.raw.response.sequences, original_values), (
        "deepcopy must preserve tensor values in raw.response.sequences"
    )
