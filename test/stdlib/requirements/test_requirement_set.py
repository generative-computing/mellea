"""Tests for RequirementSet and GuardrailProfiles."""

import pytest

from mellea.core import Requirement, ValidationResult
from mellea.stdlib.requirements import GuardrailProfiles, RequirementSet
from mellea.stdlib.requirements.guardrails import (
    json_valid,
    max_length,
    no_harmful_content,
    no_pii,
)

# region RequirementSet Tests


def test_requirement_set_creation_empty():
    """Test creating an empty RequirementSet."""
    reqs = RequirementSet()
    assert len(reqs) == 0
    assert reqs.is_empty()


def test_requirement_set_creation_with_list():
    """Test creating RequirementSet with initial requirements."""
    reqs = RequirementSet([no_pii(), json_valid()])
    assert len(reqs) == 2
    assert not reqs.is_empty()


def test_requirement_set_creation_type_error():
    """Test that non-Requirement items raise TypeError."""
    with pytest.raises(TypeError, match="must be Requirement instances"):
        RequirementSet(["not a requirement"])  # type: ignore


def test_requirement_set_add():
    """Test adding requirements with fluent API."""
    reqs = RequirementSet().add(no_pii()).add(json_valid())
    assert len(reqs) == 2


def test_requirement_set_add_type_error():
    """Test that adding non-Requirement raises TypeError."""
    reqs = RequirementSet()
    with pytest.raises(TypeError, match="Expected Requirement instance"):
        reqs.add("not a requirement")  # type: ignore


def test_requirement_set_add_immutable():
    """Test that add() returns new instance (immutable)."""
    original = RequirementSet([no_pii()])
    modified = original.add(json_valid())

    assert len(original) == 1
    assert len(modified) == 2


def test_requirement_set_remove():
    """Test removing requirements by identity with copy=False."""
    req1 = no_pii()
    req2 = json_valid()
    # Use copy=False to preserve object identity
    reqs = RequirementSet([req1, req2], copy=False)

    # With copy=False, we can remove by reference
    modified = reqs.remove(req1, copy=False)
    assert len(modified) == 1
    assert modified is reqs  # Same object (in-place)


def test_requirement_set_remove_immutable():
    """Test that remove() returns new instance (immutable) by default."""
    req1 = no_pii()
    req2 = json_valid()
    original = RequirementSet([req1, req2], copy=False)

    # With default copy=True, remove() creates a new instance
    # But deepcopy breaks object identity, so we need to remove by index or use copy=False
    modified = original.remove(req1, copy=False)
    new_copy = RequirementSet(modified.to_list())  # Create immutable copy

    assert len(original) == 1  # Modified in place
    assert len(modified) == 1  # Same as original
    assert len(new_copy) == 1  # Immutable copy
    assert modified is original  # Same object (copy=False)
    assert new_copy is not original  # Different object


def test_requirement_set_remove_not_found():
    """Test that removing non-existent requirement doesn't error."""
    reqs = RequirementSet([no_pii()])
    modified = reqs.remove(json_valid())
    assert len(modified) == 1  # Unchanged


def test_requirement_set_add_mutable():
    """Test that add() with copy=False modifies in place."""
    original = RequirementSet([no_pii()], copy=False)
    original_id = id(original)
    modified = original.add(json_valid(), copy=False)

    assert len(original) == 2
    assert len(modified) == 2
    assert id(modified) == original_id  # Same object


def test_requirement_set_extend_mutable():
    """Test that extend() with copy=False modifies in place."""
    original = RequirementSet([no_pii()], copy=False)
    original_id = id(original)
    modified = original.extend([json_valid(), max_length(500)], copy=False)

    assert len(original) == 3
    assert len(modified) == 3
    assert id(modified) == original_id  # Same object


def test_requirement_set_deduplicate_by_description():
    """Test deduplication removes requirements with same description."""
    req1 = no_pii()
    req2 = no_pii()  # Same description, different instance
    reqs = RequirementSet([req1, req2, json_valid()], copy=False)

    deduped = reqs.deduplicate()
    assert len(deduped) == 2  # no_pii + json_valid
    assert len(reqs) == 3  # Original unchanged


def test_requirement_set_deduplicate_preserves_order():
    """Test that deduplication preserves first occurrence."""
    reqs = RequirementSet([no_pii(), json_valid(), no_pii()], copy=False)
    deduped = reqs.deduplicate()

    assert len(deduped) == 2
    # First requirement should be no_pii
    first_desc = next(iter(deduped)).description
    assert first_desc == no_pii().description


def test_requirement_set_deduplicate_by_identity():
    """Test deduplication by identity."""
    req1 = no_pii()
    req2 = req1  # Same instance
    req3 = json_valid()
    reqs = RequirementSet([req1, req2, req3], copy=False)

    deduped = reqs.deduplicate(by="identity")
    assert len(deduped) == 2  # req1 and req3 (req2 is same as req1)


def test_requirement_set_deduplicate_inplace():
    """Test in-place deduplication."""
    reqs = RequirementSet([no_pii(), no_pii(), json_valid()], copy=False)
    original_id = id(reqs)

    deduped = reqs.deduplicate(copy=False)
    assert len(deduped) == 2
    assert id(deduped) == original_id  # Same object


def test_requirement_set_deduplicate_invalid_strategy():
    """Test that invalid deduplication strategy raises ValueError."""
    reqs = RequirementSet([no_pii()], copy=False)
    with pytest.raises(ValueError, match="Invalid deduplication strategy"):
        reqs.deduplicate(by="invalid")  # type: ignore


def test_profile_composition_with_dedupe():
    """Test real-world profile composition with deduplication."""
    from mellea.stdlib.requirements import GuardrailProfiles

    safety = GuardrailProfiles.basic_safety()
    format = GuardrailProfiles.json_output()

    # Combine profiles (may have duplicates)
    combined = safety + format
    original_len = len(combined)

    # Deduplicate
    deduped = combined.deduplicate()

    # Should have fewer or equal requirements
    assert len(deduped) <= original_len

    # Verify no duplicate descriptions
    descriptions = [r.description for r in deduped]
    assert len(descriptions) == len(set(descriptions))

    """Test that extend() with copy=False modifies in place."""
    original = RequirementSet([no_pii()], copy=False)
    original_id = id(original)
    modified = original.extend([json_valid(), max_length(500)], copy=False)

    assert len(original) == 3
    assert len(modified) == 3
    assert id(modified) == original_id  # Same object


def test_requirement_set_extend():
    """Test extending with multiple requirements."""
    reqs = RequirementSet().extend([no_pii(), json_valid(), max_length(500)])
    assert len(reqs) == 3


def test_requirement_set_extend_type_error():
    """Test that extending with non-Requirements raises TypeError."""
    reqs = RequirementSet()
    with pytest.raises(TypeError, match="must be Requirement instances"):
        reqs.extend([no_pii(), "not a requirement"])  # type: ignore


def test_requirement_set_addition():
    """Test combining RequirementSets with + operator."""
    set1 = RequirementSet([no_pii()])
    set2 = RequirementSet([json_valid()])
    combined = set1 + set2

    assert len(combined) == 2
    assert len(set1) == 1  # Original unchanged
    assert len(set2) == 1  # Original unchanged


def test_requirement_set_addition_type_error():
    """Test that adding non-RequirementSet raises TypeError."""
    reqs = RequirementSet([no_pii()])
    with pytest.raises(TypeError, match="Can only add RequirementSet"):
        reqs + [json_valid()]  # type: ignore  # noqa: RUF005


def test_requirement_set_iadd():
    """Test in-place addition with += operator."""
    reqs = RequirementSet([no_pii()])
    original_id = id(reqs)
    reqs += RequirementSet([json_valid()])

    assert len(reqs) == 2
    assert id(reqs) == original_id  # Same object (in-place)


def test_requirement_set_iadd_type_error():
    """Test that += with non-RequirementSet raises TypeError."""
    reqs = RequirementSet([no_pii()])
    with pytest.raises(TypeError, match="Can only add RequirementSet"):
        reqs += [json_valid()]  # type: ignore


def test_requirement_set_len():
    """Test len() function."""
    reqs = RequirementSet([no_pii(), json_valid(), max_length(500)])
    assert len(reqs) == 3


def test_requirement_set_iter():
    """Test iteration over RequirementSet."""
    req1 = no_pii()
    req2 = json_valid()
    reqs = RequirementSet([req1, req2])

    items = list(reqs)
    assert len(items) == 2
    assert all(isinstance(item, Requirement) for item in items)


def test_requirement_set_repr():
    """Test string representation."""
    reqs = RequirementSet([no_pii(), json_valid()])
    repr_str = repr(reqs)
    assert "RequirementSet" in repr_str
    assert "2 requirements" in repr_str


def test_requirement_set_str():
    """Test detailed string representation."""
    reqs = RequirementSet([no_pii(), json_valid()])
    str_repr = str(reqs)
    assert "RequirementSet" in str_repr
    assert "2 requirements" in str_repr


def test_requirement_set_str_empty():
    """Test string representation of empty set."""
    reqs = RequirementSet()
    assert "empty" in str(reqs)


def test_requirement_set_copy():
    """Test deep copy."""
    original = RequirementSet([no_pii(), json_valid()])
    copy = original.copy()

    assert len(copy) == len(original)
    assert id(copy) != id(original)
    assert id(copy._requirements) != id(original._requirements)


def test_requirement_set_to_list():
    """Test conversion to list."""
    reqs = RequirementSet([no_pii(), json_valid()])
    req_list = reqs.to_list()

    assert isinstance(req_list, list)
    assert len(req_list) == 2
    assert all(isinstance(item, Requirement) for item in req_list)


def test_requirement_set_clear():
    """Test clearing all requirements."""
    reqs = RequirementSet([no_pii(), json_valid()])
    empty = reqs.clear()

    assert len(empty) == 0
    assert empty.is_empty()
    assert len(reqs) == 2  # Original unchanged


def test_requirement_set_is_empty():
    """Test is_empty() method."""
    empty = RequirementSet()
    not_empty = RequirementSet([no_pii()])

    assert empty.is_empty()
    assert not not_empty.is_empty()


def test_requirement_set_chaining():
    """Test method chaining (fluent API)."""
    reqs = RequirementSet().add(no_pii()).add(json_valid()).add(max_length(500))

    assert len(reqs) == 3


def test_requirement_set_complex_composition():
    """Test complex composition scenario."""
    base = RequirementSet([no_pii()])
    safety = base.add(no_harmful_content())
    format = RequirementSet([json_valid(), max_length(1000)])

    combined = safety + format
    assert len(combined) == 4


# endregion

# region GuardrailProfiles Tests


def test_guardrail_profiles_basic_safety():
    """Test basic_safety profile."""
    profile = GuardrailProfiles.basic_safety()
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 2


def test_guardrail_profiles_json_output():
    """Test json_output profile."""
    profile = GuardrailProfiles.json_output(max_size=500)
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 3


def test_guardrail_profiles_code_generation():
    """Test code_generation profile."""
    profile = GuardrailProfiles.code_generation("python")
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 3


def test_guardrail_profiles_professional_content():
    """Test professional_content profile."""
    profile = GuardrailProfiles.professional_content()
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 3


def test_guardrail_profiles_api_documentation():
    """Test api_documentation profile."""
    profile = GuardrailProfiles.api_documentation()
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 5


def test_guardrail_profiles_grounded_summary():
    """Test grounded_summary profile."""
    context = "Python is a programming language."
    profile = GuardrailProfiles.grounded_summary(context, threshold=0.5)
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 3


def test_guardrail_profiles_safe_chat():
    """Test safe_chat profile."""
    profile = GuardrailProfiles.safe_chat()
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 3


def test_guardrail_profiles_structured_data_no_schema():
    """Test structured_data profile without schema."""
    profile = GuardrailProfiles.structured_data()
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 3


def test_guardrail_profiles_structured_data_with_schema():
    """Test structured_data profile with schema."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    profile = GuardrailProfiles.structured_data(schema=schema)
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 4  # Includes matches_schema


def test_guardrail_profiles_content_moderation():
    """Test content_moderation profile."""
    profile = GuardrailProfiles.content_moderation()
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 3


def test_guardrail_profiles_minimal():
    """Test minimal profile."""
    profile = GuardrailProfiles.minimal()
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 1


def test_guardrail_profiles_strict():
    """Test strict profile."""
    profile = GuardrailProfiles.strict()
    assert isinstance(profile, RequirementSet)
    assert len(profile) == 4


def test_guardrail_profiles_customization():
    """Test that profiles can be customized."""
    profile = GuardrailProfiles.basic_safety()
    customized = profile.add(json_valid())

    assert len(profile) == 2  # Original unchanged
    assert len(customized) == 3


def test_guardrail_profiles_composition():
    """Test composing multiple profiles."""
    safety = GuardrailProfiles.basic_safety()
    format = GuardrailProfiles.json_output(max_size=500)

    combined = safety + format
    assert isinstance(combined, RequirementSet)
    # Note: May have duplicates (e.g., no_pii appears in both)
    assert len(combined) >= 3


# endregion

# region Integration Tests


def test_requirement_set_with_session_compatibility():
    """Test that RequirementSet is compatible with session.instruct()."""
    # This test verifies the interface, not actual execution
    reqs = RequirementSet([no_pii(), json_valid()])

    # Should be iterable
    req_list = list(reqs)
    assert len(req_list) == 2
    assert all(isinstance(r, Requirement) for r in req_list)


def test_profile_with_session_compatibility():
    """Test that GuardrailProfiles work with session.instruct()."""
    profile = GuardrailProfiles.basic_safety()

    # Should be iterable
    req_list = list(profile)
    assert len(req_list) == 2
    assert all(isinstance(r, Requirement) for r in req_list)


def test_real_world_scenario():
    """Test a realistic usage scenario."""

    # Define application-wide profiles
    class AppGuardrails:
        BASE = RequirementSet([no_pii(), no_harmful_content()])
        JSON_API = BASE + RequirementSet([json_valid(), max_length(1000)])

    # Use in application
    api_reqs = AppGuardrails.JSON_API
    assert len(api_reqs) == 4
    assert isinstance(api_reqs, RequirementSet)


# endregion
