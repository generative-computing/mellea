# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool schemas and validation of list, dict and constrained parameters.

Regression tests for the bug where the schema rebuild for simple parameters
kept only `description`, `type`, `enum` and `default`, dropping `items`,
`additionalProperties` and constraints. The model was never told a list's
element type, and `validate_tool_arguments` (which builds its validator from
the same schema) fell back to `string` elements, so `list[int]` arguments
reached the tool as strings. The schema is shared by every backend via
`convert_tools_to_json`, and the validator runs on every backend.

See: https://github.com/generative-computing/mellea/issues/1693
"""

import json
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from mellea.backends.tools import MelleaTool, validate_tool_arguments


class Point(BaseModel):
    x: int
    y: int


class Cat(BaseModel):
    kind: Literal["cat"]
    meow: int


class Dog(BaseModel):
    kind: Literal["dog"]
    bark: int


Pet = Annotated[Cat | Dog, Field(discriminator="kind")]


class Owner(BaseModel):
    name: str
    pet: Pet


def pets_tool(pets: list[Pet]) -> int:
    """Count pets.

    Args:
        pets: the pets
    """
    return len(pets)


def optional_pets_tool(pets: list[Pet] | None = None) -> int:
    """Count pets, if given.

    Args:
        pets: the pets
    """
    return len(pets or [])


def named_pets_tool(pets: dict[str, Pet]) -> int:
    """Count named pets.

    Args:
        pets: pets by name
    """
    return len(pets)


def owners_tool(owners: list[Owner]) -> int:
    """Count owners.

    Args:
        owners: the owners
    """
    return len(owners)


def owner_tool(owner: Owner) -> str:
    """Name an owner.

    Args:
        owner: the owner
    """
    return owner.name


def ints_tool(nums: list[int]) -> int:
    """Sum some numbers.

    Args:
        nums: the numbers
    """
    return sum(nums)


def floats_tool(nums: list[float]) -> float:
    """Sum some floats.

    Args:
        nums: the numbers
    """
    return sum(nums)


def optional_ints_tool(nums: list[int] | None = None) -> int:
    """Sum some numbers, if given.

    Args:
        nums: the numbers
    """
    return sum(nums or [])


def bools_tool(flags: list[bool]) -> int:
    """Count true flags.

    Args:
        flags: the flags
    """
    return sum(flags)


def nested_ints_tool(rows: list[list[int]]) -> int:
    """Sum a matrix.

    Args:
        rows: the rows
    """
    return sum(sum(row) for row in rows)


def points_tool(pts: list[Point]) -> int:
    """Count points.

    Args:
        pts: the points
    """
    return len(pts)


def optional_points_tool(pts: list[Point] | None = None) -> int:
    """Count points, if given.

    Args:
        pts: the points
    """
    return len(pts or [])


def counts_tool(counts: dict[str, int]) -> int:
    """Total some counts.

    Args:
        counts: counts by name
    """
    return sum(counts.values())


def optional_counts_tool(counts: dict[str, int] | None = None) -> int:
    """Total some counts, if given.

    Args:
        counts: counts by name
    """
    return sum((counts or {}).values())


def bounded_tool(score: Annotated[int, Field(ge=0, le=10)]) -> int:
    """Record a score.

    Args:
        score: the score
    """
    return score


def optional_bounded_tool(score: Annotated[int, Field(ge=0)] | None = None) -> int:
    """Record a score, if given.

    Args:
        score: the score
    """
    return score or 0


def non_empty_tool(nums: Annotated[list[int], Field(min_length=1)]) -> int:
    """Sum at least one number.

    Args:
        nums: the numbers
    """
    return sum(nums)


def bounded_str_tool(
    code: Annotated[str, Field(max_length=5, pattern="^[A-Z]+$")],
) -> str:
    """Look up a code.

    Args:
        code: the code
    """
    return code


def any_list_tool(values: list[Any]) -> int:
    """Count values.

    Args:
        values: the values
    """
    return len(values)


def described_any_list_tool(
    values: list[Annotated[Any, Field(description="a value")]],
) -> int:
    """Count values.

    Args:
        values: the values
    """
    return len(values)


def strings_tool(words: list[str]) -> int:
    """Count words.

    Args:
        words: the words
    """
    return len(words)


def _prop(func, param):
    """Return the serialized schema for one parameter of a callable's tool."""
    return MelleaTool.from_callable(func).as_json_tool["function"]["parameters"][
        "properties"
    ][param]


def _required(func):
    """Return the `required` list of a callable's tool schema."""
    params = MelleaTool.from_callable(func).as_json_tool["function"]["parameters"]
    return params.get("required", [])


# ============================================================================
# Schema generation: what every backend sends to the model
# ============================================================================


class TestElementTypesInSchema:
    """Element types of lists and dicts must reach the model."""

    def test_list_int_property_shape(self):
        assert _prop(ints_tool, "nums") == {
            "description": "the numbers",
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_list_float_items(self):
        assert _prop(floats_tool, "nums")["items"] == {"type": "number"}

    def test_list_bool_items(self):
        assert _prop(bools_tool, "flags")["items"] == {"type": "boolean"}

    def test_list_str_items(self):
        assert _prop(strings_tool, "words")["items"] == {"type": "string"}

    def test_nested_list_items(self):
        assert _prop(nested_ints_tool, "rows")["items"] == {
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_optional_list_items_taken_from_array_branch(self):
        prop = _prop(optional_ints_tool, "nums")
        assert prop["type"] == "array"
        assert prop["items"] == {"type": "integer"}
        assert "nums" not in _required(optional_ints_tool)

    def test_list_of_models_items_inlined(self):
        items = _prop(points_tool, "pts")["items"]
        assert "$ref" not in items
        assert items["type"] == "object"
        assert set(items["properties"]) == {"x", "y"}
        assert items["properties"]["x"]["type"] == "integer"
        assert set(items["required"]) == {"x", "y"}

    def test_optional_list_of_models_items_inlined(self):
        prop = _prop(optional_points_tool, "pts")
        assert prop["type"] == "array"
        assert "$ref" not in prop["items"]
        assert set(prop["items"]["properties"]) == {"x", "y"}
        assert "pts" not in _required(optional_points_tool)

    def test_dict_value_type(self):
        prop = _prop(counts_tool, "counts")
        assert prop["type"] == "object"
        assert prop["additionalProperties"] == {"type": "integer"}

    def test_optional_dict_value_type_taken_from_branch(self):
        prop = _prop(optional_counts_tool, "counts")
        assert prop["type"] == "object"
        assert prop["additionalProperties"] == {"type": "integer"}
        assert "counts" not in _required(optional_counts_tool)

    def test_any_list_keeps_unconstrained_items(self):
        assert _prop(any_list_tool, "values")["items"] == {}


class TestDiscriminatedUnionsInCollections:
    """Discriminated unions in carried keywords are flattened for tool APIs."""

    @pytest.mark.parametrize(
        ("func", "param"),
        [
            (pets_tool, "pets"),
            (optional_pets_tool, "pets"),
            (named_pets_tool, "pets"),
            (owners_tool, "owners"),
            (owner_tool, "owner"),
        ],
        ids=["list", "optional_list", "dict", "list_of_models", "model_field"],
    )
    def test_no_oneof_or_discriminator(self, func, param):
        rendered = json.dumps(_prop(func, param))
        assert "oneOf" not in rendered
        assert "discriminator" not in rendered
        assert "$ref" not in rendered

    def test_list_items_become_inlined_anyof(self):
        items = _prop(pets_tool, "pets")["items"]
        kinds = {b["properties"]["kind"]["const"] for b in items["anyOf"]}
        assert kinds == {"cat", "dog"}

    def test_dict_values_become_inlined_anyof(self):
        values = _prop(named_pets_tool, "pets")["additionalProperties"]
        assert len(values["anyOf"]) == 2


class TestConstraintsInSchema:
    """`Field` constraints must reach the model."""

    def test_numeric_bounds(self):
        prop = _prop(bounded_tool, "score")
        assert prop["type"] == "integer"
        assert prop["minimum"] == 0
        assert prop["maximum"] == 10

    def test_optional_numeric_bound_taken_from_branch(self):
        prop = _prop(optional_bounded_tool, "score")
        assert prop["type"] == "integer"
        assert prop["minimum"] == 0

    def test_list_length(self):
        prop = _prop(non_empty_tool, "nums")
        assert prop["minItems"] == 1
        assert prop["items"] == {"type": "integer"}

    def test_string_length_and_pattern(self):
        prop = _prop(bounded_str_tool, "code")
        assert prop["maxLength"] == 5
        assert prop["pattern"] == "^[A-Z]+$"

    def test_title_not_carried(self):
        """Pydantic's per-property `title` stays stripped, as before."""
        assert "title" not in _prop(bounded_tool, "score")
        assert "title" not in _prop(ints_tool, "nums")


# ============================================================================
# Validation: what the tool actually receives
# ============================================================================


class TestListValidation:
    """Correct list arguments must pass validation with their element types intact."""

    def test_list_int_not_coerced_to_str(self):
        tool = MelleaTool.from_callable(ints_tool)
        validated = validate_tool_arguments(tool, {"nums": [1, 2]}, strict=True)
        assert validated == {"nums": [1, 2]}
        assert all(type(n) is int for n in validated["nums"])

    def test_list_int_tool_runs(self):
        """The reproducer from the issue: the tool must not raise TypeError."""
        tool = MelleaTool.from_callable(ints_tool)
        validated = validate_tool_arguments(tool, {"nums": [1, 2]}, strict=True)
        assert tool.run(**validated) == 3

    def test_list_int_elements_coerced_from_str(self):
        tool = MelleaTool.from_callable(ints_tool)
        validated = validate_tool_arguments(tool, {"nums": ["1", "2"]})
        assert validated == {"nums": [1, 2]}

    def test_list_float(self):
        tool = MelleaTool.from_callable(floats_tool)
        validated = validate_tool_arguments(tool, {"nums": [1.5]}, strict=True)
        assert validated == {"nums": [1.5]}
        assert type(validated["nums"][0]) is float

    def test_optional_list_int(self):
        tool = MelleaTool.from_callable(optional_ints_tool)
        validated = validate_tool_arguments(tool, {"nums": [1, 2]}, strict=True)
        assert validated == {"nums": [1, 2]}
        assert all(type(n) is int for n in validated["nums"])

    def test_optional_list_int_none(self):
        tool = MelleaTool.from_callable(optional_ints_tool)
        validated = validate_tool_arguments(tool, {"nums": None}, strict=True)
        assert validated == {"nums": None}

    def test_optional_dict_int(self):
        tool = MelleaTool.from_callable(optional_counts_tool)
        validated = validate_tool_arguments(tool, {"counts": {"a": 1}}, strict=True)
        assert validated == {"counts": {"a": 1}}
        assert type(validated["counts"]["a"]) is int

    def test_optional_dict_int_none(self):
        tool = MelleaTool.from_callable(optional_counts_tool)
        validated = validate_tool_arguments(tool, {"counts": None}, strict=True)
        assert validated == {"counts": None}

    @pytest.mark.parametrize(
        ("func", "args"),
        [
            (bools_tool, {"flags": [True, False]}),
            (nested_ints_tool, {"rows": [[1], [2, 3]]}),
            (points_tool, {"pts": [{"x": 1, "y": 2}]}),
            (optional_points_tool, {"pts": [{"x": 1, "y": 2}]}),
        ],
        ids=["list_bool", "list_list_int", "list_model", "optional_list_model"],
    )
    def test_previously_rejected_types_validate(self, func, args):
        """These used to skip validation; `strict=True` proves they now pass it."""
        tool = MelleaTool.from_callable(func)
        assert validate_tool_arguments(tool, args, strict=True) == args

    def test_list_of_models_rejects_missing_field(self):
        """Element validation is real: a point missing `y` is rejected."""
        tool = MelleaTool.from_callable(points_tool)
        with pytest.raises(ValidationError, match=r"pts\.0\.y"):
            validate_tool_arguments(tool, {"pts": [{"x": 1}]}, strict=True)

    def test_list_str_still_coerces_numbers(self):
        """Existing `list[str]` coercion of numbers to strings is unchanged."""
        tool = MelleaTool.from_callable(strings_tool)
        validated = validate_tool_arguments(tool, {"words": ["a", 1]})
        assert validated == {"words": ["a", "1"]}

    def test_any_list_elements_untouched(self):
        tool = MelleaTool.from_callable(any_list_tool)
        validated = validate_tool_arguments(
            tool, {"values": [1, "a", True]}, strict=True
        )
        assert validated == {"values": [1, "a", True]}
        assert type(validated["values"][0]) is int

    def test_list_of_discriminated_union_validates(self):
        """Elements are validated against the branch the tag selects."""
        tool = MelleaTool.from_callable(pets_tool)
        validated = validate_tool_arguments(
            tool, {"pets": [{"kind": "cat", "meow": "1"}]}, strict=True
        )
        assert validated == {"pets": [{"kind": "cat", "meow": 1}]}

    def test_list_of_discriminated_union_rejects_wrong_branch_fields(self):
        tool = MelleaTool.from_callable(pets_tool)
        with pytest.raises(ValidationError, match=r"pets\.0\.cat\.meow"):
            validate_tool_arguments(
                tool, {"pets": [{"kind": "cat", "bark": 1}]}, strict=True
            )

    def test_described_any_list_elements_untouched(self):
        """An annotation-only items schema constrains nothing either."""
        tool = MelleaTool.from_callable(described_any_list_tool)
        validated = validate_tool_arguments(tool, {"values": [1, 2]}, strict=True)
        assert validated == {"values": [1, 2]}
        assert type(validated["values"][0]) is int

    @pytest.mark.parametrize(
        "array_schema",
        [
            {"type": "array"},
            {"type": "array", "items": True},
            {"type": "array", "items": {"description": "a value"}},
        ],
        ids=["no_items", "items_true", "annotation_only_items"],
    )
    def test_external_unconstrained_array_elements_untouched(self, array_schema):
        """Unconstrained external `items` must not coerce elements to strings."""
        as_json_tool = {
            "type": "function",
            "function": {
                "name": "external",
                "description": "An externally defined tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"values": array_schema},
                    "required": ["values"],
                },
            },
        }
        tool = MelleaTool("external", lambda values: values, as_json_tool)
        validated = validate_tool_arguments(tool, {"values": [1, 2.5]}, strict=True)
        assert validated == {"values": [1, 2.5]}
        assert type(validated["values"][0]) is int
