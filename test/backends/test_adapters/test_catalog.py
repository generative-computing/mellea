"""Unit tests for the intrinsics catalog — metadata lookup, validation, and enumeration."""

import pytest

from mellea.backends.adapters.catalog import (
    AdapterType,
    fetch_intrinsic_metadata,
    known_intrinsic_names,
)

# --- known_intrinsic_names ---


def test_known_intrinsic_names_returns_non_empty_list():
    names = known_intrinsic_names()
    assert isinstance(names, list)
    assert len(names) > 0


def test_known_intrinsic_names_contains_expected_entries():
    names = known_intrinsic_names()
    for expected in ("answerability", "citations", "uncertainty"):
        assert expected in names


# --- fetch_intrinsic_metadata ---


def test_fetch_returns_correct_entry():
    entry = fetch_intrinsic_metadata("answerability")
    assert entry.name == "answerability"
    assert isinstance(entry.repo_id, str)
    assert len(entry.repo_id) > 0


def test_fetch_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown intrinsic name 'bogus'"):
        fetch_intrinsic_metadata("bogus")


def test_fetch_returns_defensive_copy():
    entry_a = fetch_intrinsic_metadata("answerability")
    entry_b = fetch_intrinsic_metadata("answerability")
    assert entry_a == entry_b
    assert entry_a is not entry_b


# --- adapter types ---


def test_default_adapter_types():
    entry = fetch_intrinsic_metadata("answerability")
    assert AdapterType.LORA in entry.adapter_types
    assert AdapterType.ALORA in entry.adapter_types


def test_lora_only_entry():
    entry = fetch_intrinsic_metadata("query_clarification")
    assert entry.adapter_types == (AdapterType.LORA,)
