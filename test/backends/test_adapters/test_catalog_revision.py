"""Tests for IntriniscsCatalogEntry revision field and requirement_check deduplication."""

import pydantic
import pytest

from mellea.backends.adapters.catalog import (
    _INTRINSICS_CATALOG_ENTRIES,
    IntriniscsCatalogEntry,
    fetch_intrinsic_metadata,
    known_intrinsic_names,
    validate_revision,
)

_VALID_SHA = "a" * 40
_SHORT_SHA = "a" * 39
_LONG_SHA = "a" * 41
_NONHEX_SHA = "g" * 40
_UPPER_SHA = "A" * 40


def test_catalog_entries_have_revision():
    for entry in _INTRINSICS_CATALOG_ENTRIES:
        rev = entry.revision
        assert rev == "main" or (
            len(rev) == 40
            and rev == rev.lower()
            and all(c in "0123456789abcdef" for c in rev)
        ), f"entry {entry.name!r} has invalid revision {rev!r}"


def test_revision_validation_rejects_malformed():
    for bad in [_SHORT_SHA, _LONG_SHA, _NONHEX_SHA, _UPPER_SHA, "", "HEAD", "latest"]:
        with pytest.raises(ValueError):
            validate_revision(bad)


def test_revision_validation_accepts_valid_sha():
    assert validate_revision(_VALID_SHA) == _VALID_SHA


def test_revision_validation_accepts_main_literal():
    assert validate_revision("main") == "main"


def test_revision_field_rejects_malformed_via_pydantic():
    with pytest.raises(pydantic.ValidationError):
        IntriniscsCatalogEntry(name="x", repo_id="org/repo", revision=_SHORT_SHA)


def test_revision_field_rejects_none_via_pydantic():
    with pytest.raises(pydantic.ValidationError):
        IntriniscsCatalogEntry(name="x", repo_id="org/repo", revision=None)  # type: ignore[arg-type]


def test_revision_round_trip():
    entry = IntriniscsCatalogEntry(name="x", repo_id="org/repo", revision=_VALID_SHA)
    assert entry.revision == _VALID_SHA

    entry_main = IntriniscsCatalogEntry(name="y", repo_id="org/repo", revision="main")
    assert entry_main.revision == "main"


def test_revision_round_trip_via_fetch():
    entry = fetch_intrinsic_metadata("answerability")
    rev = entry.revision
    assert len(rev) == 40 and rev == rev.lower()


def test_no_duplicate_requirement_check_entry():
    names = known_intrinsic_names()
    # Only the underscore variant should be present.
    assert "requirement_check" in names
    assert "requirement-check" not in names
    # Exactly one entry with underscore.
    matching = [e for e in _INTRINSICS_CATALOG_ENTRIES if e.name == "requirement_check"]
    assert len(matching) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
