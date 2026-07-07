"""Unit tests for the intrinsics catalog — metadata lookup, validation, and enumeration."""

import pytest

from mellea.backends.adapters.catalog import (
    _INTRINSICS_CATALOG_ENTRIES,
    AdapterType,
    IntrinsicsCatalogEntry,
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


def test_lora_only_entry(monkeypatch):
    from mellea.backends.adapters import catalog

    fake_entry = catalog.IntrinsicsCatalogEntry(
        name="query_clarification",
        repo_id="ibm-granite/granitelib-rag-r1.0",
        revision="main",
        adapter_types=(AdapterType.LORA,),
    )
    monkeypatch.setattr(
        catalog, "_INTRINSICS_CATALOG", {"query_clarification": fake_entry}
    )
    entry = fetch_intrinsic_metadata("query_clarification")
    assert entry.adapter_types == (AdapterType.LORA,)


# --- capability / effective_capability ---


def test_effective_capability_defaults_to_name():
    entry = fetch_intrinsic_metadata("answerability")
    assert entry.capability is None
    assert entry.effective_capability == "answerability"


def test_effective_capability_returns_explicit_capability():
    entry = fetch_intrinsic_metadata("requirement-check")
    assert entry.effective_capability == "requirement_check"


@pytest.mark.parametrize(
    "name,expected_capability",
    [
        ("context-attribution", "context_attribution"),
        ("requirement-check", "requirement_check"),
        ("policy-guardrails", "policy_guardrails"),
        ("guardian-core", "guardian_core"),
        ("factuality-detection", "factuality_detection"),
        ("factuality-correction", "factuality_correction"),
    ],
)
def test_hyphenated_entries_have_underscore_capabilities(name, expected_capability):
    entry = fetch_intrinsic_metadata(name)
    assert entry.capability == expected_capability
    assert entry.effective_capability == expected_capability


@pytest.mark.parametrize(
    "name",
    [
        "answerability",
        "citations",
        "context_relevance",
        "hallucination_detection",
        "query_clarification",
        "query_rewrite",
        "uncertainty",
    ],
)
def test_clean_named_entries_have_no_explicit_capability(name):
    entry = fetch_intrinsic_metadata(name)
    assert entry.capability is None
    assert entry.effective_capability == name


def test_effective_capability_not_in_model_dump():
    entry = fetch_intrinsic_metadata("requirement-check")
    dumped = entry.model_dump()
    assert "effective_capability" not in dumped
    assert "capability" in dumped
    assert dumped["capability"] == "requirement_check"


def test_capability_field_used_when_constructing_directly():
    entry = IntrinsicsCatalogEntry(
        name="my-custom-adapter",
        capability="my_custom_adapter",
        repo_id="org/repo",
        revision="abc123def456",
    )
    assert entry.effective_capability == "my_custom_adapter"


def test_capability_none_means_effective_capability_equals_name():
    entry = IntrinsicsCatalogEntry(
        name="plain_name", repo_id="org/repo", revision="abc123def456"
    )
    assert entry.capability is None
    assert entry.effective_capability == "plain_name"


def test_all_catalog_effective_capabilities_are_nonempty():
    for entry in _INTRINSICS_CATALOG_ENTRIES:
        assert entry.effective_capability, (
            f"Entry {entry.name!r} has an empty effective_capability"
        )


def test_name_empty_string_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        IntrinsicsCatalogEntry(name="", repo_id="org/repo", revision="abc123")


def test_name_whitespace_only_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        IntrinsicsCatalogEntry(name="   ", repo_id="org/repo", revision="abc123")


def test_name_leading_trailing_whitespace_rejected():
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        IntrinsicsCatalogEntry(name="  foo  ", repo_id="org/repo", revision="abc123")


def test_capability_empty_string_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        IntrinsicsCatalogEntry(
            name="x", capability="", repo_id="org/repo", revision="abc123"
        )


def test_capability_whitespace_only_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        IntrinsicsCatalogEntry(
            name="x", capability="   ", repo_id="org/repo", revision="abc123"
        )


def test_capability_leading_trailing_whitespace_rejected():
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        IntrinsicsCatalogEntry(
            name="x", capability="  foo  ", repo_id="org/repo", revision="abc123"
        )
