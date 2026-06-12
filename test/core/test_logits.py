"""Unit tests for ModelOutputThunk.logits field."""

import copy

import pytest

from mellea.core import ModelOutputThunk


def test_mot_logits_defaults_to_none():
    mot = ModelOutputThunk(value="hello")
    assert mot.logits is None


def test_mot_logits_can_be_set():
    fake_scores = (object(), object())
    mot = ModelOutputThunk(value="hello")
    mot.logits = fake_scores
    assert mot.logits is fake_scores


def test_mot_logits_preserved_by_copy():
    fake_scores = (object(), object())
    mot = ModelOutputThunk(value="hello")
    mot.logits = fake_scores
    copied = copy.copy(mot)
    assert copied.logits is fake_scores


def test_mot_logits_preserved_by_deepcopy():
    fake_scores = ([1, 2, 3], [4, 5, 6])
    mot = ModelOutputThunk(value="hello")
    mot.logits = fake_scores
    deepcopied = copy.deepcopy(mot)
    assert deepcopied.logits == fake_scores
    assert deepcopied.logits is not fake_scores


if __name__ == "__main__":
    pytest.main([__file__])
