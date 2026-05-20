"""End-to-end tests for discriminated-union tool parameters.

Covers issue #989: a tool parameter typed as a Pydantic discriminated union
``Annotated[A | B, Field(discriminator="kind")]`` (with or without ``| None``)
must not collapse to ``{"type": "string"}``. The schema produced by
``convert_function_to_ollama_tool`` is consumed by every backend
(Ollama, OpenAI, Watsonx, HuggingFace, LiteLLM), so the union structure must
be preserved and the OAS-3 ``discriminator`` keyword must be stripped from
the output (the JSON Schema subset accepted by tool-calling APIs does not
include it; the ``Literal`` tag fields carry the discriminator signal).
"""

from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from mellea.backends.tools import (
    MelleaTool,
    convert_function_to_ollama_tool,
    validate_tool_arguments,
)


class Cat(BaseModel):
    kind: Literal["cat"]
    name: str


class Dog(BaseModel):
    kind: Literal["dog"]
    name: str
    breed: str


def act(pet: Annotated[Cat | Dog, Field(discriminator="kind")]) -> str:
    """Act on a pet.

    Args:
        pet: the pet to act on
    """
    return "ok"


def act_optional(
    pet: Annotated[Cat | Dog, Field(discriminator="kind")] | None = None,
) -> str:
    """Act on an optional pet.

    Args:
        pet: the pet to act on, may be omitted
    """
    return "ok"


def _pet_schema(func) -> dict:
    """Convert ``func`` and return the ``pet`` parameter schema."""
    tool = convert_function_to_ollama_tool(func)
    return tool.function.parameters.model_dump(exclude_none=True)["properties"]["pet"]


def _has_branch(schema: dict, kind_value: str, *, must_have: set[str]) -> bool:
    """Check that ``schema`` contains an inlined object branch for ``kind_value``."""
    branches = schema.get("anyOf") or schema.get("oneOf") or []
    for branch in branches:
        props = branch.get("properties", {})
        kind = props.get("kind", {})
        if kind.get("const") == kind_value or kind_value in (kind.get("enum") or []):
            return must_have.issubset(set(props.keys()))
    return False


class TestDiscriminatedUnionSchema:
    """Schema-shape assertions for discriminated-union tool parameters."""

    def test_required_union_does_not_collapse_to_string(self):
        """The discriminated union must not be flattened to a primitive."""
        pet = _pet_schema(act)
        assert pet.get("type") != "string", (
            f"discriminated union collapsed to a string schema: {pet!r}"
        )

    def test_required_union_preserves_branches(self):
        """Both Cat and Dog branches must survive as inlined object schemas."""
        pet = _pet_schema(act)
        assert "anyOf" in pet or "oneOf" in pet, f"expected anyOf/oneOf in {pet!r}"
        assert _has_branch(pet, "cat", must_have={"kind", "name"}), (
            f"Cat branch missing or unresolved: {pet!r}"
        )
        assert _has_branch(pet, "dog", must_have={"kind", "name", "breed"}), (
            f"Dog branch missing or unresolved: {pet!r}"
        )

    def test_required_union_strips_discriminator_keyword(self):
        """OAS-3 ``discriminator`` is rejected by Ollama / OpenAI strict mode.

        The ``Literal`` constraint on ``kind`` already carries the tag signal,
        so the OAS keyword adds no semantic value but is actively harmful.
        """
        pet = _pet_schema(act)
        assert "discriminator" not in pet, (
            f"discriminator keyword should be stripped from output: {pet!r}"
        )

    def test_required_union_no_dangling_refs(self):
        """No ``$ref`` should leak into the output for the issue reproducer."""
        import json

        rendered = json.dumps(_pet_schema(act))
        assert "$ref" not in rendered, f"unresolved $ref in tool schema: {rendered}"

    def test_optional_union_does_not_collapse_to_string(self):
        """The Optional variant also must not flatten to a primitive."""
        pet = _pet_schema(act_optional)
        # Either pet is itself a discriminated union schema with a null branch,
        # or it is anyOf:[<union>, null]. Either way, "string" alone is wrong.
        assert pet.get("type") != "string", (
            f"optional discriminated union collapsed to a string schema: {pet!r}"
        )

    def test_optional_union_preserves_branches(self):
        """The Optional variant must preserve both inlined object branches."""
        pet = _pet_schema(act_optional)
        assert _has_branch(pet, "cat", must_have={"kind", "name"}), (
            f"Cat branch missing in optional variant: {pet!r}"
        )
        assert _has_branch(pet, "dog", must_have={"kind", "name", "breed"}), (
            f"Dog branch missing in optional variant: {pet!r}"
        )

    def test_optional_union_drops_from_required(self):
        """The optional parameter must not be in the function's required list."""
        tool = convert_function_to_ollama_tool(act_optional)
        params = tool.function.parameters.model_dump(exclude_none=True)
        assert "pet" not in params.get("required", []), (
            f"optional 'pet' should not be required: {params}"
        )


class TestDiscriminatedUnionValidation:
    """``validate_tool_arguments`` must round-trip a valid discriminated payload."""

    def test_strict_accepts_valid_dog(self):
        """A correctly-shaped dog dict should pass strict validation."""
        mt = MelleaTool.from_callable(act)
        validate_tool_arguments(
            mt, {"pet": {"kind": "dog", "name": "Rex", "breed": "lab"}}, strict=True
        )

    def test_strict_accepts_valid_cat(self):
        """A correctly-shaped cat dict should pass strict validation."""
        mt = MelleaTool.from_callable(act)
        validate_tool_arguments(
            mt, {"pet": {"kind": "cat", "name": "Whiskers"}}, strict=True
        )

    def test_strict_rejects_bare_string(self):
        """A bare string was the bug's silent-pass: must now be rejected."""
        mt = MelleaTool.from_callable(act)
        with pytest.raises(ValidationError):
            validate_tool_arguments(mt, {"pet": "just a string"}, strict=True)

    def test_strict_rejects_missing_discriminator(self):
        """A dict without the ``kind`` discriminator must be rejected."""
        mt = MelleaTool.from_callable(act)
        with pytest.raises(ValidationError):
            validate_tool_arguments(mt, {"pet": {"name": "Rex"}}, strict=True)

    def test_optional_accepts_omitted(self):
        """The optional variant accepts the parameter being omitted."""
        mt = MelleaTool.from_callable(act_optional)
        validate_tool_arguments(mt, {}, strict=True)

    def test_optional_accepts_valid_payload(self):
        """The optional variant accepts a valid payload."""
        mt = MelleaTool.from_callable(act_optional)
        validate_tool_arguments(
            mt, {"pet": {"kind": "dog", "name": "Rex", "breed": "lab"}}, strict=True
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
