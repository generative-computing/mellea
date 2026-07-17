# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GenerationMetadata.logits field on ModelOutputThunk."""

import copy

from mellea.core import ModelOutputThunk


def test_mot_logits_defaults_to_none():
    mot = ModelOutputThunk(value="hello")
    assert mot.generation.logits is None


def test_mot_logits_can_be_set():
    fake_scores = (object(), object())
    mot = ModelOutputThunk(value="hello")
    mot.generation.logits = fake_scores
    assert mot.generation.logits is fake_scores


def test_mot_logits_preserved_by_copy():
    fake_scores = (object(), object())
    mot = ModelOutputThunk(value="hello")
    mot.generation.logits = fake_scores
    copied = copy.copy(mot)
    assert copied.generation.logits is fake_scores


def test_mot_logits_preserved_by_deepcopy():
    fake_scores = ([1, 2, 3], [4, 5, 6])
    mot = ModelOutputThunk(value="hello")
    mot.generation.logits = fake_scores
    deepcopied = copy.deepcopy(mot)
    assert deepcopied.generation.logits == fake_scores
    assert deepcopied.generation.logits is not fake_scores
