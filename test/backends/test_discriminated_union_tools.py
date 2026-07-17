# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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

import json
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


class Fish(BaseModel):
    kind: Literal["fish"]
    name: str
    species: str


class Email(BaseModel):
    """Non-discriminated nested model for the no-op regression test."""

    to: str
    subject: str


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
    assert tool.function is not None
    assert tool.function.parameters is not None
    return tool.function.parameters.model_dump(exclude_none=True)["properties"]["pet"]


def _has_branch(schema: dict, kind_value: str, *, must_have: set[str]) -> bool:
    """Check that ``schema`` contains an inlined ``anyOf`` branch for ``kind_value``.

    After the fix lands the output schema must contain ``anyOf`` only, never
    ``oneOf`` — accepting ``oneOf`` here would silently mask a regression of
    the discriminator-flattening pre-pass.
    """
    branches = schema.get("anyOf", [])
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
        assert pet.get("description") == "the pet to act on", (
            f"docstring description lost during flattening: {pet!r}"
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
        assert tool.function is not None
        assert tool.function.parameters is not None
        params = tool.function.parameters.model_dump(exclude_none=True)
        assert "pet" not in params.get("required", []), (
            f"optional 'pet' should not be required: {params}"
        )

    def test_optional_union_strips_discriminator_keyword(self):
        """The Optional variant must also drop the OAS-3 ``discriminator``.

        The required variant strips it via the top-level ``oneOf`` path; the
        optional variant strips it implicitly when the wrapper sub-schema is
        replaced by its expanded branches. Asserted explicitly so a refactor
        that re-introduces the wrapper does not slip past silently.
        """
        rendered = json.dumps(_pet_schema(act_optional))
        assert "discriminator" not in rendered, (
            f"discriminator keyword should be stripped from optional output: {rendered}"
        )

    def test_three_way_union_preserves_all_branches(self):
        """A three-arm discriminated union must preserve all three branches."""

        def act_three(
            pet: Annotated[Cat | Dog | Fish, Field(discriminator="kind")],
        ) -> str:
            """Act on a three-way pet.

            Args:
                pet: the pet to act on
            """
            return "ok"

        pet = _pet_schema(act_three)
        assert _has_branch(pet, "cat", must_have={"kind", "name"}), (
            f"Cat branch missing in three-way union: {pet!r}"
        )
        assert _has_branch(pet, "dog", must_have={"kind", "name", "breed"}), (
            f"Dog branch missing in three-way union: {pet!r}"
        )
        assert _has_branch(pet, "fish", must_have={"kind", "name", "species"}), (
            f"Fish branch missing in three-way union: {pet!r}"
        )

    def test_non_discriminated_optional_unchanged(self):
        """Non-discriminated ``Optional[Email]`` must still flow through unchanged.

        Regression guard: the new pre-pass must be a no-op for plain
        ``$ref`` + ``| None`` shapes that the existing inliner already
        handles. Pydantic emits this as
        ``{"anyOf": [{"$ref": "..."}, {"type": "null"}]}`` — no ``oneOf``
        in any sub-schema, so the pre-pass should not activate.
        """

        def send(email: Email | None = None) -> str:
            """Send an email.

            Args:
                email: optional email payload
            """
            return "sent"

        tool = convert_function_to_ollama_tool(send)
        assert tool.function is not None
        assert tool.function.parameters is not None
        rendered = tool.function.parameters.model_dump(exclude_none=True)
        email_schema = rendered["properties"]["email"]
        # The existing complex-anyOf path inlines the $ref and preserves the
        # full object schema with properties. The exact shape is owned by the
        # pre-existing logic; we only assert the pre-pass did not collapse it.
        assert email_schema.get("type") != "string", (
            f"non-discriminated Optional collapsed: {email_schema!r}"
        )
        assert "email" not in rendered.get("required", []), (
            f"optional email should not be required: {rendered}"
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
