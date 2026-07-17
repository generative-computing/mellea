# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for IntrinsicsCatalogEntry revision field and requirement-check deduplication."""

import pydantic
import pytest

from mellea.backends.adapters.catalog import (
    IntrinsicsCatalogEntry,
    fetch_intrinsic_metadata,
    known_intrinsic_names,
    validate_revision,
)

_VALID_SHA = "a" * 40


def test_catalog_entries_have_revision():
    for name in known_intrinsic_names():
        rev = fetch_intrinsic_metadata(name).revision
        # Raises ValueError if the entry's revision is empty.
        validate_revision(rev)


def test_revision_validation_rejects_empty():
    with pytest.raises(ValueError):
        validate_revision("")


def test_revision_validation_rejects_whitespace():
    with pytest.raises(ValueError):
        validate_revision("   ")


def test_revision_validation_rejects_leading_trailing_whitespace():
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        validate_revision("  main  ")


def test_revision_validation_accepts_sha():
    assert validate_revision(_VALID_SHA) == _VALID_SHA


def test_revision_validation_accepts_branch_and_tag():
    # HuggingFace's revision parameter takes a branch name, tag, or commit SHA;
    # the validator mirrors that contract.
    for rev in ["main", "dev", "v1.0", "release-2026-05"]:
        assert validate_revision(rev) == rev


def test_revision_field_rejects_empty_via_pydantic():
    with pytest.raises(pydantic.ValidationError):
        IntrinsicsCatalogEntry(name="x", repo_id="org/repo", revision="")


def test_revision_field_rejects_none_via_pydantic():
    with pytest.raises(pydantic.ValidationError):
        IntrinsicsCatalogEntry(name="x", repo_id="org/repo", revision=None)  # type: ignore[arg-type]


def test_revision_round_trip():
    entry = IntrinsicsCatalogEntry(name="x", repo_id="org/repo", revision=_VALID_SHA)
    assert entry.revision == _VALID_SHA

    entry_main = IntrinsicsCatalogEntry(name="y", repo_id="org/repo", revision="main")
    assert entry_main.revision == "main"


def test_revision_round_trip_via_fetch():
    entry = fetch_intrinsic_metadata("answerability")
    assert entry.revision  # non-empty


def test_no_duplicate_requirement_check_entry():
    names = known_intrinsic_names()
    # Only the hyphen variant should be present — it matches the folder layout
    # in ibm-granite/granitelib-core-r1.0.
    assert "requirement-check" in names
    assert "requirement_check" not in names
    # Exactly one entry with the hyphen.
    assert names.count("requirement-check") == 1
