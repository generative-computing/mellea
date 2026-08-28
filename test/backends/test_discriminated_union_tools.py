# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for discriminated-union tool parameters.

Covers issue #989: a tool parameter typed as a Pydantic discriminated union
`Annotated[A | B, Field(discriminator="kind")]` (with or without `| None`)
must not collapse to `{"type": "string"}`. The schema produced by
`convert_function_to_ollama_tool` is consumed by every backend
(Ollama, OpenAI, Watsonx, HuggingFace, LiteLLM), so the union structure must
be preserved and the OAS-3 `discriminator` keyword must be stripped from
the output (the JSON Schema subset accepted by tool-calling APIs does not
include it; the `Literal` tag fields carry the discriminator signal).

Also covers `convert_function_to_ollama_tool`'s handling of postponed
annotations (`from __future__ import annotations`, PEP 563): non-builtin
parameter types must resolve to real type objects for Pydantic schema
building, while a return annotation that is unresolvable at call time (e.g.
`TYPE_CHECKING`-only imports) must not break the conversion, since the
produced schema never consumes it.
"""

import functools
import inspect
import json
from dataclasses import dataclass
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, PydanticUserError, ValidationError

from mellea.backends.tools import (
    MelleaTool,
    convert_function_to_ollama_tool,
    validate_tool_arguments,
)
from test.backends._postponed_annotation_samples import (
    Address,
    Period,
    Zone as samples_zone,
    send_letter,
    tc_only_return_builtin_param,
    tc_return_custom_param,
    tc_return_region,
    tc_return_unannotated_param,
    tc_return_zone,
    unresolvable_param,
)


@dataclass
class Zone:
    """Deliberately shadows `Zone` in `_postponed_annotation_samples`.

    Binding this name in the test module's globals is what makes
    `test_decorated_tool_resolves_colliding_type_name` meaningful: a wrapper
    defined here resolves against these globals under the buggy behaviour, so
    the wrong `Zone` is found and no error is raised. The field name differs
    from the real one so the resulting schema is distinguishable.
    """

    wrong_field: int


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
    """Convert `func` and return the `pet` parameter schema."""
    tool = convert_function_to_ollama_tool(func)
    assert tool.function is not None
    assert tool.function.parameters is not None
    return tool.function.parameters.model_dump(exclude_none=True)["properties"]["pet"]


def _has_branch(schema: dict, kind_value: str, *, must_have: set[str]) -> bool:
    """Check that `schema` contains an inlined `anyOf` branch for `kind_value`.

    After the fix lands the output schema must contain `anyOf` only, never
    `oneOf` — accepting `oneOf` here would silently mask a regression of
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
        """OAS-3 `discriminator` is rejected by Ollama / OpenAI strict mode.

        The `Literal` constraint on `kind` already carries the tag signal,
        so the OAS keyword adds no semantic value but is actively harmful.
        """
        pet = _pet_schema(act)
        assert "discriminator" not in pet, (
            f"discriminator keyword should be stripped from output: {pet!r}"
        )

    def test_required_union_no_dangling_refs(self):
        """No `$ref` should leak into the output for the issue reproducer."""
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
        """The Optional variant must also drop the OAS-3 `discriminator`.

        The required variant strips it via the top-level `oneOf` path; the
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
        """Non-discriminated `Optional[Email]` must still flow through unchanged.

        Regression guard: the new pre-pass must be a no-op for plain
        `$ref` + `| None` shapes that the existing inliner already
        handles. Pydantic emits this as
        `{"anyOf": [{"$ref": "..."}, {"type": "null"}]}` — no `oneOf`
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
    """`validate_tool_arguments` must round-trip a valid discriminated payload."""

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
        """A dict without the `kind` discriminator must be rejected."""
        mt = MelleaTool.from_callable(act)
        with pytest.raises(ValidationError):
            validate_tool_arguments(mt, {"pet": {"name": "Rex"}}, strict=True)

    def test_strict_rejects_invalid_discriminator_value(self):
        """A kind value outside the allowed set must be rejected."""
        mt = MelleaTool.from_callable(act)
        with pytest.raises(ValidationError):
            validate_tool_arguments(
                mt,
                {"pet": {"kind": "horse", "name": "Bob", "breed": "lab"}},
                strict=True,
            )

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


class TestNestedDiscriminatedUnions:
    """Tests for nested discriminated unions (unions within union branches)."""

    def test_nested_discriminated_union_flattens(self):
        """Nested discriminated unions must be recursively flattened.

        When a tool parameter has a discriminated union whose branches
        themselves contain discriminated unions, all levels must be flattened:
        - All oneOf converted to anyOf
        - All discriminator keywords stripped
        - All $ref entries within nested discriminated unions inlined
        """

        class FullUser(BaseModel):
            type: Literal["full"]
            name: str
            email: str

        class StubUser(BaseModel):
            type: Literal["stub"]
            user_id: str

        class CreateUserFull(BaseModel):
            op: Literal["create_full"]
            user_data: Annotated[FullUser | StubUser, Field(discriminator="type")]

        class CreateUserStub(BaseModel):
            op: Literal["create_stub"]
            user_id: str

        class DeleteUser(BaseModel):
            op: Literal["delete"]
            user_id: str

        def execute(
            cmd: Annotated[
                CreateUserFull | CreateUserStub | DeleteUser, Field(discriminator="op")
            ],
        ) -> str:
            """Execute a command.

            Args:
                cmd: the command to execute
            """
            return "ok"

        tool = convert_function_to_ollama_tool(execute)
        assert tool.function is not None
        assert tool.function.parameters is not None
        params = tool.function.parameters.model_dump(exclude_none=True)
        cmd_schema = params["properties"]["cmd"]

        # Check that outer union is flattened
        assert "anyOf" in cmd_schema, "outer union must be in anyOf"
        assert "oneOf" not in cmd_schema, "outer union must not have oneOf"

        # Check each branch for nested discriminated unions
        for branch in cmd_schema.get("anyOf", []):
            if branch.get("title") == "CreateUserFull":
                # This branch has a nested discriminated union
                user_data = branch.get("properties", {}).get("user_data", {})

                # The nested union must be flattened too
                assert "anyOf" in user_data, (
                    f"nested discriminated union must be in anyOf: {user_data}"
                )
                assert "oneOf" not in user_data, (
                    f"nested union must not have oneOf: {user_data}"
                )
                assert "discriminator" not in user_data, (
                    f"nested union must not have discriminator keyword: {user_data}"
                )

                # All branches of the nested union must be inlined objects
                nested_branches = user_data.get("anyOf", [])
                assert len(nested_branches) == 2, "nested union should have 2 branches"
                for nested_branch in nested_branches:
                    # Should be inlined object, not a $ref
                    assert "$ref" not in nested_branch, (
                        f"nested branch should not have $ref: {nested_branch}"
                    )
                    assert "properties" in nested_branch, (
                        f"nested branch should be inlined object: {nested_branch}"
                    )
                    assert nested_branch.get("type") == "object", (
                        f"nested branch should be object type: {nested_branch}"
                    )

    def test_deeply_nested_discriminated_unions(self):
        """Discriminated unions at 3+ levels of nesting must all be flattened."""

        class Level3A(BaseModel):
            l3_type: Literal["a"]
            value: str

        class Level3B(BaseModel):
            l3_type: Literal["b"]
            value: int

        class Level2A(BaseModel):
            l2_type: Literal["a"]
            nested: Annotated[Level3A | Level3B, Field(discriminator="l3_type")]

        class Level2B(BaseModel):
            l2_type: Literal["b"]
            text: str

        class Level1A(BaseModel):
            l1_type: Literal["a"]
            nested: Annotated[Level2A | Level2B, Field(discriminator="l2_type")]

        class Level1B(BaseModel):
            l1_type: Literal["b"]
            name: str

        def deeply_nested(
            param: Annotated[Level1A | Level1B, Field(discriminator="l1_type")],
        ) -> str:
            """Deeply nested discriminated unions.

            Args:
                param: the parameter
            """
            return "ok"

        tool = convert_function_to_ollama_tool(deeply_nested)
        assert tool.function is not None
        assert tool.function.parameters is not None
        params = tool.function.parameters.model_dump(exclude_none=True)
        param_schema = params["properties"]["param"]

        # Render entire schema to JSON to search for problematic patterns
        rendered = json.dumps(param_schema)

        # No oneOf should remain anywhere in the schema
        assert "oneOf" not in rendered, (
            "All levels of nesting should have oneOf flattened to anyOf"
        )

        # No discriminator keyword should remain anywhere
        assert "discriminator" not in rendered, (
            "All discriminator keywords should be stripped"
        )

        # Verify at least level 1 and level 2 are properly structured
        level1_branches = param_schema.get("anyOf", [])
        for branch in level1_branches:
            if branch.get("title") == "Level1A":
                # Has nested Level2
                nested = branch.get("properties", {}).get("nested", {})
                assert "anyOf" in nested, "Level 2 should use anyOf"
                assert "oneOf" not in nested, "Level 2 should not have oneOf"

                # Check Level 2 branches
                level2_branches = nested.get("anyOf", [])
                for level2_branch in level2_branches:
                    if level2_branch.get("title") == "Level2A":
                        # Has nested Level3
                        level3_nested = level2_branch.get("properties", {}).get(
                            "nested", {}
                        )
                        assert "anyOf" in level3_nested, "Level 3 should use anyOf"
                        assert "oneOf" not in level3_nested, (
                            "Level 3 should not have oneOf"
                        )

    def test_nested_union_no_refs_leak(self):
        """No $ref should leak into the nested discriminated union schema."""

        class Inner1(BaseModel):
            inner_type: Literal["inner1"]
            value: str

        class Inner2(BaseModel):
            inner_type: Literal["inner2"]
            count: int

        class Outer(BaseModel):
            outer_type: Literal["outer"]
            inner: Annotated[Inner1 | Inner2, Field(discriminator="inner_type")]

        def cmd(
            param: Annotated[Outer, Field(discriminator="outer_type")] | None = None,
        ) -> str:
            """Command.

            Args:
                param: optional outer parameter
            """
            return "ok"

        tool = convert_function_to_ollama_tool(cmd)
        assert tool.function is not None
        assert tool.function.parameters is not None
        params = tool.function.parameters.model_dump(exclude_none=True)

        # Check the entire parameter schema for leaked $refs
        rendered = json.dumps(params["properties"]["param"])
        assert "$ref" not in rendered, (
            f"No $ref should leak into the final schema: {rendered[:200]}..."
        )

    def test_nested_union_payload_round_trips(self):
        """Valid nested discriminated union payloads must round-trip through validate_tool_arguments."""

        class FullUser(BaseModel):
            type: Literal["full"]
            name: str
            email: str

        class StubUser(BaseModel):
            type: Literal["stub"]
            user_id: str

        class CreateUserFull(BaseModel):
            op: Literal["create_full"]
            user_data: Annotated[FullUser | StubUser, Field(discriminator="type")]

        class DeleteUser(BaseModel):
            op: Literal["delete"]
            user_id: str

        def execute(
            cmd: Annotated[CreateUserFull | DeleteUser, Field(discriminator="op")],
        ) -> str:
            """Execute a command.

            Args:
                cmd: the command to execute
            """
            return "ok"

        mt = MelleaTool.from_callable(execute)

        # Test valid full user creation
        validate_tool_arguments(
            mt,
            {
                "cmd": {
                    "op": "create_full",
                    "user_data": {
                        "type": "full",
                        "name": "Ada",
                        "email": "ada@example.com",
                    },
                }
            },
            strict=True,
        )

        # Test valid stub user creation
        validate_tool_arguments(
            mt,
            {
                "cmd": {
                    "op": "create_full",
                    "user_data": {"type": "stub", "user_id": "user123"},
                }
            },
            strict=True,
        )

        # Test delete command
        validate_tool_arguments(
            mt, {"cmd": {"op": "delete", "user_id": "user456"}}, strict=True
        )

    def test_nested_union_payload_rejects_invalid_discriminator(self):
        """Invalid nested discriminator values must be rejected."""

        class FullUser(BaseModel):
            type: Literal["full"]
            name: str
            email: str

        class StubUser(BaseModel):
            type: Literal["stub"]
            user_id: str

        class CreateUserFull(BaseModel):
            op: Literal["create_full"]
            user_data: Annotated[FullUser | StubUser, Field(discriminator="type")]

        class DeleteUser(BaseModel):
            op: Literal["delete"]
            user_id: str

        def execute(
            cmd: Annotated[CreateUserFull | DeleteUser, Field(discriminator="op")],
        ) -> str:
            """Execute a command.

            Args:
                cmd: the command to execute
            """
            return "ok"

        mt = MelleaTool.from_callable(execute)

        # Invalid nested discriminator value
        with pytest.raises(ValidationError):
            validate_tool_arguments(
                mt,
                {
                    "cmd": {
                        "op": "create_full",
                        "user_data": {
                            "type": "invalid",
                            "name": "Ada",
                            "email": "ada@example.com",
                        },
                    }
                },
                strict=True,
            )

    def test_nested_union_payload_rejects_missing_nested_field(self):
        """Missing required fields in nested union branches must be rejected."""

        class FullUser(BaseModel):
            type: Literal["full"]
            name: str
            email: str

        class CreateUserFull(BaseModel):
            op: Literal["create_full"]
            user_data: Annotated[FullUser, Field(discriminator="type")]

        def execute(cmd: Annotated[CreateUserFull, Field(discriminator="op")]) -> str:
            """Execute a command.

            Args:
                cmd: the command to execute
            """
            return "ok"

        mt = MelleaTool.from_callable(execute)

        # Missing required 'email' field in nested FullUser
        with pytest.raises(ValidationError):
            validate_tool_arguments(
                mt,
                {
                    "cmd": {
                        "op": "create_full",
                        "user_data": {"type": "full", "name": "Ada"},
                    }
                },
                strict=True,
            )

    def test_deeply_nested_union_payload_round_trips(self):
        """Deeply nested discriminated union payloads must round-trip through validate_tool_arguments."""

        class Level3A(BaseModel):
            l3_type: Literal["a"]
            value: str

        class Level3B(BaseModel):
            l3_type: Literal["b"]
            value: int

        class Level2A(BaseModel):
            l2_type: Literal["a"]
            nested: Annotated[Level3A | Level3B, Field(discriminator="l3_type")]

        class Level2B(BaseModel):
            l2_type: Literal["b"]
            text: str

        class Level1A(BaseModel):
            l1_type: Literal["a"]
            nested: Annotated[Level2A | Level2B, Field(discriminator="l2_type")]

        class Level1B(BaseModel):
            l1_type: Literal["b"]
            name: str

        def deeply_nested(
            param: Annotated[Level1A | Level1B, Field(discriminator="l1_type")],
        ) -> str:
            """Deeply nested discriminated unions.

            Args:
                param: the parameter
            """
            return "ok"

        mt = MelleaTool.from_callable(deeply_nested)

        # Test payload with three levels of nesting
        validate_tool_arguments(
            mt,
            {
                "param": {
                    "l1_type": "a",
                    "nested": {
                        "l2_type": "a",
                        "nested": {"l3_type": "a", "value": "deep_value"},
                    },
                }
            },
            strict=True,
        )

        # Test alternative path through nesting
        validate_tool_arguments(
            mt,
            {
                "param": {
                    "l1_type": "a",
                    "nested": {"l2_type": "a", "nested": {"l3_type": "b", "value": 42}},
                }
            },
            strict=True,
        )

        # Test simplified path
        validate_tool_arguments(
            mt,
            {"param": {"l1_type": "a", "nested": {"l2_type": "b", "text": "simple"}}},
            strict=True,
        )

        # Test Level1B branch
        validate_tool_arguments(
            mt, {"param": {"l1_type": "b", "name": "direct"}}, strict=True
        )


class TestPostponedAnnotations:
    """Postponed annotation (PEP 563) resolution in generated tool schemas."""

    def test_resolves_postponed_parameter_annotation(self):
        # Regression test: under `from __future__ import annotations`, a
        # non-builtin parameter type's annotation is a string rather than the
        # real type object, which Pydantic cannot resolve when building the
        # dynamic schema model - raising PydanticUserError instead of producing
        # a tool schema.

        # Guard the precondition: if the sample module ever drops its
        # `from __future__ import annotations`, this test would otherwise keep
        # passing without exercising postponed annotations at all.
        assert send_letter.__annotations__["to"] == "Address"

        tool = convert_function_to_ollama_tool(send_letter)
        assert tool.function is not None
        assert tool.function.parameters is not None

        props = tool.function.parameters.model_dump(exclude_none=True)["properties"]
        assert props["to"]["type"] == "object"
        assert props["to"]["title"] == Address.__name__
        assert props["to"]["properties"]["city"]["type"] == "string"

    def test_type_checking_only_return_with_builtin_params(self):
        """TYPE_CHECKING-only return + builtin params must produce schema."""
        # Guard the precondition: without this, a resolved `Decimal` return
        # annotation would make the test pass trivially, without exercising
        # the unresolvable-return fallback at all.
        assert tc_only_return_builtin_param.__annotations__["return"] == "Decimal"
        # A postponed string is necessary but not sufficient: if `Decimal` ever
        # moves out of the sample module's `if TYPE_CHECKING:` block, the string
        # would still be postponed but `eval_str=True` would resolve it and the
        # fallback would never be taken.
        with pytest.raises(NameError):
            inspect.signature(tc_only_return_builtin_param, eval_str=True)

        tool = convert_function_to_ollama_tool(tc_only_return_builtin_param)
        assert tool.function is not None
        assert tool.function.parameters is not None
        props = tool.function.parameters.model_dump(exclude_none=True)["properties"]
        assert "query" in props

    def test_type_checking_only_return_with_custom_param(self):
        """TYPE_CHECKING return + custom param must resolve param.

        This is the case that separates the try/except-around-`eval_str=True`
        fallback (which discards parameter resolution entirely on any
        failure) from per-parameter resolution: the return annotation is
        unresolvable, but the custom parameter type still must resolve.
        """
        # Guard the precondition: both must still be postponed strings, or
        # this test stops exercising the per-parameter resolution path.
        assert tc_return_custom_param.__annotations__["return"] == "Decimal"
        assert tc_return_custom_param.__annotations__["period"] == "Period"
        # And, as above, the fallback must be the path under test rather than
        # `eval_str=True` succeeding outright.
        with pytest.raises(NameError):
            inspect.signature(tc_return_custom_param, eval_str=True)

        tool = convert_function_to_ollama_tool(tc_return_custom_param)
        assert tool.function is not None
        assert tool.function.parameters is not None
        props = tool.function.parameters.model_dump(exclude_none=True)["properties"]
        assert props["period"]["type"] == "object"
        assert props["period"]["title"] == Period.__name__

    def test_unannotated_parameter_passes_through_fallback(self):
        """A bare parameter must survive the per-parameter loop untouched.

        In the fallback path an unannotated parameter is `inspect._empty`, not a
        string, so it takes the branch that skips resolution. It should still
        reach the schema, defaulting to `str` as it does on the normal path.
        """
        # Guard the precondition: the fallback must be the path under test.
        with pytest.raises(NameError):
            inspect.signature(tc_return_unannotated_param, eval_str=True)

        tool = convert_function_to_ollama_tool(tc_return_unannotated_param)
        assert tool.function is not None
        assert tool.function.parameters is not None
        props = tool.function.parameters.model_dump(exclude_none=True)["properties"]
        assert props["flag"]["type"] == "string"
        assert props["period"]["type"] == "object"
        assert props["period"]["title"] == Period.__name__

    def test_unresolvable_parameter_annotation_raises(self):
        """Genuinely unresolvable parameter must still raise PydanticUserError.

        Confirms the documented degradation: unresolvable param annotations
        surface as `PydanticUserError`, not `NameError`.
        """
        # Guard the precondition: if this were ever resolved, the test would
        # pass without exercising the unresolvable-parameter fallback.
        assert unresolvable_param.__annotations__["query"] == "NonExistentType"

        with pytest.raises(PydanticUserError, match="NonExistentType"):
            convert_function_to_ollama_tool(unresolvable_param)

    def test_decorated_tool_resolves_in_wrapped_functions_module(self):
        """A `functools.wraps` wrapper must not shift the resolution namespace.

        `inspect.signature` follows `__wrapped__`, so the annotations being
        resolved belong to the wrapped function's module. Resolving them
        against the decorator's module instead finds the wrong type, or none
        at all - and the failure is silent, producing a wrong tool schema
        rather than an error.
        """

        # The wrapper must be defined here, not in the sample module: its
        # `__globals__` is the namespace the buggy code would have used, and it
        # has to be one that lacks `Region`. Moving it beside `tc_return_region`
        # would make the test pass either way.
        @functools.wraps(tc_return_region)
        def wrapper(*args, **kwargs):
            return tc_return_region(*args, **kwargs)

        # This module deliberately does not import `Region`, so resolving
        # against this module's namespace cannot find it. That is what makes
        # the assertion below meaningful.
        assert "Region" not in globals()

        tool = convert_function_to_ollama_tool(wrapper)
        assert tool.function is not None
        assert tool.function.parameters is not None
        props = tool.function.parameters.model_dump(exclude_none=True)["properties"]
        assert props["region"]["type"] == "object"
        assert props["region"]["title"] == "Region"
        assert props["region"]["properties"]["code"]["type"] == "string"

    def test_decorated_tool_resolves_colliding_type_name(self):
        """A colliding type name must resolve to the wrapped function's type.

        The sibling test above covers a name *absent* from the decorator's
        module, which degrades loudly to `PydanticUserError`. This covers the
        worse case: the name exists in the decorator's module but refers to a
        different type, so resolving in the wrong namespace succeeds and emits
        a plausible-looking schema for entirely the wrong type.
        """

        # Defined here, so this module's globals - which bind `Zone` to the
        # shadowing class above - are what the buggy path would resolve against.
        @functools.wraps(tc_return_zone)
        def wrapper(*args, **kwargs):
            return tc_return_zone(*args, **kwargs)

        # Guard the preconditions: the annotation must still be postponed, the
        # fallback must be the path under test, and the two `Zone` classes must
        # genuinely differ - otherwise the assertions below prove nothing.
        assert tc_return_zone.__annotations__["zone"] == "Zone"
        with pytest.raises(NameError):
            inspect.signature(wrapper, eval_str=True)
        assert globals()["Zone"] is not samples_zone
        assert "wrong_field" in globals()["Zone"].__dataclass_fields__

        tool = convert_function_to_ollama_tool(wrapper)
        assert tool.function is not None
        assert tool.function.parameters is not None
        props = tool.function.parameters.model_dump(exclude_none=True)["properties"]
        assert props["zone"]["properties"] == {
            "identifier": {"title": "Identifier", "type": "string"}
        }
        assert "wrong_field" not in props["zone"]["properties"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
