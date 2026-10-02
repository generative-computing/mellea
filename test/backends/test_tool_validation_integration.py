# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the validate_tool_arguments function.

These tests verify that the validation function works correctly with
the actual tool call flow.
"""

from typing import Any, Optional, Union

import pytest
from pydantic import ValidationError

from mellea.backends.tools import MelleaTool, validate_tool_arguments
from mellea.core import ModelToolCall

# ============================================================================
# Test Fixtures - Tool Functions
# ============================================================================


def simple_tool(message: str) -> str:
    """A simple tool that takes a string.

    Args:
        message: The message to process
    """
    return f"Processed: {message}"


def typed_tool(name: str, age: int, score: float, active: bool) -> dict:
    """Tool with multiple primitive types.

    Args:
        name: Person's name
        age: Person's age in years
        score: Performance score
        active: Whether person is active
    """
    return {"name": name, "age": age, "score": score, "active": active}


def optional_tool(required: str, optional: str | None = None) -> str:
    """Tool with optional parameters.

    Args:
        required: A required parameter
        optional: An optional parameter
    """
    return f"{required}:{optional or 'none'}"


def limit_tool(count: int, limit: int | None = None) -> int:
    """Tool with a required int and an optional int.

    Args:
        count: How many items to return
        limit: An optional upper bound
    """
    return count if limit is None else min(count, limit)


def paged_tool(query: str, page_size: int = 10) -> str:
    """Tool with a non-nullable defaulted parameter.

    Args:
        query: The search query
        page_size: Results per page
    """
    return f"{query}:{page_size}"


def union_tool(value: str | int) -> str:
    """Tool with union type parameter.

    Args:
        value: Can be string or integer
    """
    return f"Value: {value} (type: {type(value).__name__})"


def list_tool(items: list[str]) -> int:
    """Tool with list parameter.

    Args:
        items: List of string items
    """
    return len(items)


def dict_tool(config: dict[str, Any]) -> str:
    """Tool with dict parameter.

    Args:
        config: Configuration dictionary
    """
    import json

    return json.dumps(config)


def no_params_tool() -> str:
    """Tool with no parameters."""
    return "No params needed"


def untyped_param(message) -> str:
    """A tool with an untyped parameter.

    Args:
        message: The message to process (no type hint)
    """
    return f"Processed: {message}"


# ============================================================================
# Test Cases: Type Coercion
# ============================================================================


class TestTypeCoercion:
    """Test automatic type coercion with validation."""

    def test_string_to_int_coercion(self):
        """Test that string "30" is coerced to int 30."""
        args = {"name": "Test", "age": "30", "score": 95.5, "active": True}
        tool = MelleaTool.from_callable(typed_tool)
        validated = validate_tool_arguments(tool, args, coerce_types=True)

        assert validated["age"] == 30
        assert isinstance(validated["age"], int)

    def test_string_to_float_coercion(self):
        """Test that string "95.5" is coerced to float 95.5."""
        args = {"name": "Test", "age": 30, "score": "95.5", "active": True}
        tool = MelleaTool.from_callable(typed_tool)
        validated = validate_tool_arguments(tool, args, coerce_types=True)

        assert validated["score"] == 95.5
        assert isinstance(validated["score"], float)

    def test_int_to_float_coercion(self):
        """Test that int 95 is coerced to float 95.0."""
        args = {"name": "Test", "age": 30, "score": 95, "active": True}
        tool = MelleaTool.from_callable(typed_tool)
        validated = validate_tool_arguments(tool, args, coerce_types=True)

        assert validated["score"] == 95.0
        assert isinstance(validated["score"], float)

    def test_int_to_string_coercion(self):
        """Test that int 123 is coerced to string "123"."""
        args = {"message": 123}
        tool = MelleaTool.from_callable(simple_tool)
        validated = validate_tool_arguments(tool, args, coerce_types=True)

        assert validated["message"] == "123"
        assert isinstance(validated["message"], str)

    def test_bool_coercion_from_int(self):
        """Test that int 1/0 is coerced to bool True/False."""
        args = {"name": "Test", "age": 30, "score": 95.5, "active": 1}
        tool = MelleaTool.from_callable(typed_tool)
        validated = validate_tool_arguments(tool, args, coerce_types=True)

        assert validated["active"] is True
        assert isinstance(validated["active"], bool)

        args["active"] = 0
        validated = validate_tool_arguments(tool, args, coerce_types=True)
        assert validated["active"] is False


class TestValidationModes:
    """Test strict vs. lenient validation modes."""

    def test_lenient_mode_with_invalid_type(self):
        """Test that lenient mode returns original args on validation failure."""
        args = {"name": "Test", "age": "not_a_number", "score": 95.5, "active": True}
        tool = MelleaTool.from_callable(typed_tool)
        validated = validate_tool_arguments(tool, args, strict=False)

        # Should return original args
        assert validated == args
        assert validated["age"] == "not_a_number"

    def test_strict_mode_with_invalid_type(self):
        """Test that strict mode raises ValidationError on failure."""
        args = {"name": "Test", "age": "not_a_number", "score": 95.5, "active": True}
        tool = MelleaTool.from_callable(typed_tool)

        with pytest.raises(ValidationError):
            validate_tool_arguments(tool, args, strict=True)

    def test_lenient_mode_with_missing_required(self):
        """Test lenient mode with missing required parameter."""
        args = {"optional": "value"}  # Missing 'required'
        tool = MelleaTool.from_callable(optional_tool)
        validated = validate_tool_arguments(tool, args, strict=False)

        # Should return original args
        assert validated == args

    def test_strict_mode_with_missing_required(self):
        """Test strict mode with missing required parameter."""
        args = {"optional": "value"}  # Missing 'required'
        tool = MelleaTool.from_callable(optional_tool)

        with pytest.raises(ValidationError):
            validate_tool_arguments(tool, args, strict=True)


class TestWithModelToolCall:
    """Test validation integrated with ModelToolCall."""

    def test_validated_tool_call_with_coercion(self):
        """Test that validated args work correctly with ModelToolCall."""
        # LLM returns age as string
        args = {"name": "Alice", "age": "30", "score": "95.5", "active": True}
        tool = MelleaTool.from_callable(typed_tool)

        # Validate and coerce
        validated_args = validate_tool_arguments(tool, args, coerce_types=True)

        # Create tool call with validated args
        tool_call = ModelToolCall("typed_tool", tool, validated_args)
        result = tool_call.call_func()

        # Verify result has correct types
        assert result["age"] == 30
        assert isinstance(result["age"], int)
        assert result["score"] == 95.5
        assert isinstance(result["score"], float)

    def test_unvalidated_vs_validated_comparison(self):
        """Compare behavior with and without validation."""
        args = {"name": "Bob", "age": "25", "score": "88.7", "active": True}
        tool = MelleaTool.from_callable(typed_tool)

        # Without validation - types stay as strings
        unvalidated_call = ModelToolCall("typed_tool", tool, args)
        unvalidated_result = unvalidated_call.call_func()
        assert isinstance(unvalidated_result["age"], str)  # Still string!

        # With validation - types are coerced
        validated_args = validate_tool_arguments(tool, args, coerce_types=True)
        validated_call = ModelToolCall("typed_tool", tool, validated_args)
        validated_result = validated_call.call_func()
        assert isinstance(validated_result["age"], int)  # Correctly coerced!


class TestOptionalParameters:
    """Test validation with optional parameters."""

    def test_optional_param_provided(self):
        """Test validation when optional parameter is provided."""
        args = {"required": "value1", "optional": "value2"}
        tool = MelleaTool.from_callable(optional_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated == args

    def test_optional_param_omitted(self):
        """Test validation when optional parameter is omitted."""
        args = {"required": "value1"}
        tool = MelleaTool.from_callable(optional_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated["required"] == "value1"
        # An omitted optional field must NOT be padded back in as None.
        assert "optional" not in validated

    def test_optional_param_none(self):
        """Explicit None for `str | None` validates (strict, so no fallback)."""
        args = {"required": "value1", "optional": None}
        tool = MelleaTool.from_callable(optional_tool)
        validated = validate_tool_arguments(tool, args, strict=True)

        assert validated == {"required": "value1", "optional": None}

    def test_optional_int_none_strict(self):
        """An explicit None for `int | None` validates."""
        args = {"count": 3, "limit": None}
        tool = MelleaTool.from_callable(limit_tool)
        validated = validate_tool_arguments(tool, args, strict=True)

        assert validated == {"count": 3, "limit": None}

    def test_null_for_non_nullable_default_rejected_strict(self):
        """A real default (`page_size: int = 10`) still rejects None."""
        args = {"query": "q", "page_size": None}
        tool = MelleaTool.from_callable(paged_tool)
        with pytest.raises(ValidationError, match="page_size"):
            validate_tool_arguments(tool, args, strict=True)

    def test_optional_none_keeps_other_coercions(self):
        """A None must not make lenient mode drop the other coercions."""
        args = {"count": "3", "limit": None}
        tool = MelleaTool.from_callable(limit_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated == {"count": 3, "limit": None}
        assert type(validated["count"]) is int


class TestDefaultedParameters:
    """Test that omitted defaulted parameters fall through to Python defaults."""

    def test_omitted_default_not_materialized_as_none(self):
        """An omitted defaulted parameter must not be forced to None.

        Materializing it as None would override the callable's own default
        (e.g. weather(location="London") reaching the function as units=None).
        """

        def weather(location: str, units: str = "celsius", days: int = 1) -> dict:
            """Get weather.

            Args:
                location: City name
                units: Temperature units
                days: Number of days
            """
            return {"location": location, "units": units, "days": days}

        tool = MelleaTool.from_callable(weather)
        validated = validate_tool_arguments(tool, {"location": "London"})

        assert validated == {"location": "London"}
        assert "units" not in validated
        assert "days" not in validated

        # The Python defaults take effect when the callable actually runs.
        result = ModelToolCall("weather", tool, validated).call_func()
        assert result == {"location": "London", "units": "celsius", "days": 1}

    def test_supplied_default_param_preserved(self):
        """A defaulted parameter that IS supplied is kept (and coerced)."""

        def weather(location: str, days: int = 1) -> dict:
            """Get weather.

            Args:
                location: City name
                days: Number of days
            """
            return {"location": location, "days": days}

        tool = MelleaTool.from_callable(weather)
        validated = validate_tool_arguments(
            tool, {"location": "London", "days": "3"}, coerce_types=True
        )

        assert validated["days"] == 3
        assert isinstance(validated["days"], int)


class TestComplexTypes:
    """Test validation with complex types."""

    def test_list_parameter(self):
        """Test validation with list parameter."""
        args = {"items": ["apple", "banana", "cherry"]}
        tool = MelleaTool.from_callable(list_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated["items"] == ["apple", "banana", "cherry"]
        assert isinstance(validated["items"], list)

    def test_dict_parameter(self):
        """Test validation with dict parameter."""
        args = {"config": {"key1": "value1", "key2": 42, "key3": True}}
        tool = MelleaTool.from_callable(dict_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated["config"] == args["config"]
        assert isinstance(validated["config"], dict)

    def test_empty_list(self):
        """Test validation with empty list."""
        args = {"items": []}
        tool = MelleaTool.from_callable(list_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated["items"] == []


class TestUnionTypes:
    """Test validation with union types."""

    def test_union_with_string(self):
        """Test union type with string value."""
        args = {"value": "hello"}
        tool = MelleaTool.from_callable(union_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated["value"] == "hello"
        assert isinstance(validated["value"], str)

    def test_union_with_int(self):
        """Test union type with integer value."""
        args = {"value": 42}
        tool = MelleaTool.from_callable(union_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated["value"] == 42
        assert isinstance(validated["value"], int)

    def test_union_with_string_number(self):
        """Test union type with string that looks like number."""
        args = {"value": "42"}
        tool = MelleaTool.from_callable(union_tool)
        validated = validate_tool_arguments(tool, args, coerce_types=True)

        # Pydantic will try to coerce to the first matching type
        # Result depends on Union order and Pydantic's coercion rules
        assert validated["value"] in ["42", 42]


def _external_tool(value_schema: Any) -> MelleaTool:
    """Build a tool whose single `value` parameter uses a raw JSON schema."""
    as_json_tool = {
        "type": "function",
        "function": {
            "name": "external",
            "description": "An externally defined tool.",
            "parameters": {
                "type": "object",
                "properties": {"value": value_schema},
                "required": ["value"],
            },
        },
    }
    return MelleaTool("external", lambda value: value, as_json_tool)


# Boolean sub-schemas are valid JSON Schema (e.g. from MCP) but broke the validator.
BOOLEAN_SUBSCHEMAS = [
    pytest.param(True, 1, id="property_true"),
    pytest.param({"type": "object", "properties": {"a": True}}, {"a": 1}, id="nested"),
    pytest.param({"anyOf": [True, {"type": "null"}]}, 1, id="anyof"),
]


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.parametrize(("value_schema", "value"), BOOLEAN_SUBSCHEMAS)
    def test_unsupported_schema_lenient_returns_original_args(
        self, value_schema, value
    ):
        """An unbuildable schema falls back to the original args, not a raise."""
        tool = _external_tool(value_schema)
        assert validate_tool_arguments(tool, {"value": value}) == {"value": value}

    @pytest.mark.parametrize(("value_schema", "value"), BOOLEAN_SUBSCHEMAS)
    def test_unsupported_schema_strict_raises(self, value_schema, value):
        tool = _external_tool(value_schema)
        with pytest.raises((TypeError, AttributeError)):
            validate_tool_arguments(tool, {"value": value}, strict=True)

    def test_no_parameters_tool(self):
        """Test validation with no-parameter tool."""
        args = {}
        tool = MelleaTool.from_callable(no_params_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated == {}

    def test_no_parameters_with_extra_args(self):
        """Test that extra args for no-param tool are handled."""
        args = {"fake_param": "should_be_ignored"}
        tool = MelleaTool.from_callable(no_params_tool)

        # In lenient mode, returns original args
        validated = validate_tool_arguments(tool, args, strict=False)
        assert validated == args

        # In strict mode, should raise
        with pytest.raises(ValidationError):
            validate_tool_arguments(tool, args, strict=True)

    def test_whitespace_stripping(self):
        """Test that whitespace is stripped from strings."""
        args = {"message": "  hello world  "}
        tool = MelleaTool.from_callable(simple_tool)
        validated = validate_tool_arguments(tool, args, coerce_types=True)

        assert validated["message"] == "hello world"

    def test_empty_string(self):
        """Test validation with empty string."""
        args = {"message": ""}
        tool = MelleaTool.from_callable(simple_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated["message"] == ""


class TestErrorMessages:
    """Test that error messages are helpful."""

    def test_missing_required_error_message(self):
        """Test error message for missing required parameter."""
        args = {}
        tool = MelleaTool.from_callable(simple_tool)

        try:
            validate_tool_arguments(tool, args, strict=True)
            pytest.fail("Should have raised ValidationError")
        except ValidationError as e:
            error_str = str(e)
            assert "message" in error_str.lower()
            assert "required" in error_str.lower() or "missing" in error_str.lower()

    def test_type_mismatch_error_message(self):
        """Test error message for type mismatch."""
        args = {"name": "Test", "age": "not_a_number", "score": 95.5, "active": True}
        tool = MelleaTool.from_callable(typed_tool)

        try:
            validate_tool_arguments(tool, args, strict=True)
            pytest.fail("Should have raised ValidationError")
        except ValidationError as e:
            error_str = str(e)
            assert "age" in error_str.lower()


class TestUntypedParameters:
    """Test validation with untyped parameters."""

    def test_untyped_parameter_accepts_string(self):
        """Test that untyped parameters accept string values."""
        args = {"message": "test"}
        tool = MelleaTool.from_callable(untyped_param)
        validated = validate_tool_arguments(tool, args)

        assert validated["message"] == "test"

    def test_untyped_parameter_accepts_int(self):
        """Test that untyped parameters accept integer values.

        Note: Without type hints, validation may coerce to string for safety.
        """
        args = {"message": 123}
        tool = MelleaTool.from_callable(untyped_param)
        validated = validate_tool_arguments(tool, args)

        # Validation may coerce to string when no type hint is present
        assert validated["message"] in [123, "123"]

    def test_untyped_parameter_accepts_dict(self):
        """Test untyped parameter with complex type (dict)."""
        args = {"message": {"key": "value", "number": 42}}
        tool = MelleaTool.from_callable(untyped_param)
        validated = validate_tool_arguments(tool, args)

        assert validated["message"] == {"key": "value", "number": 42}

    def test_untyped_parameter_accepts_list(self):
        """Test untyped parameter with list."""
        args = {"message": ["item1", "item2", "item3"]}
        tool = MelleaTool.from_callable(untyped_param)
        validated = validate_tool_arguments(tool, args)

        assert validated["message"] == ["item1", "item2", "item3"]

    def test_untyped_parameter_accepts_bool(self):
        """Test untyped parameter with boolean."""
        args = {"message": True}
        tool = MelleaTool.from_callable(untyped_param)
        validated = validate_tool_arguments(tool, args)

        assert validated["message"] is True

    def test_untyped_parameter_accepts_none(self):
        """Test untyped parameter with None."""
        args = {"message": None}
        tool = MelleaTool.from_callable(untyped_param)
        validated = validate_tool_arguments(tool, args)

        assert validated["message"] is None

    def test_untyped_parameter_no_coercion(self):
        """Test that untyped parameters don't get coerced."""
        args = {"message": "123"}
        tool = MelleaTool.from_callable(untyped_param)
        validated = validate_tool_arguments(tool, args, coerce_types=True)

        # Should remain as string since there's no type hint to coerce to
        assert validated["message"] == "123"
        assert isinstance(validated["message"], str)


class TestNoNullPadding:
    """Validation must return only what the model sent, never padding unset fields.

    A bare `model_dump()` used to emit a `None` for every declared-but-unset optional
    field, at every nesting depth, inventing arguments the model never produced.
    `model_dump(exclude_unset=True)` fixes this; these tests lock the behavior in.
    """

    def _nested_object_tool(self) -> MelleaTool:
        """Build a tool with a nested object whose sub-fields are all optional.

        Constructed directly from an explicit JSON schema (rather than
        `from_callable`) so the nested object can declare fields the model
        will not send.
        """
        schema = {
            "type": "function",
            "function": {
                "name": "update_profile",
                "description": "",
                "parameters": {
                    "type": "object",
                    "required": ["user_id", "settings"],
                    "properties": {
                        "user_id": {"type": "string"},
                        "settings": {
                            "type": "object",
                            "properties": {
                                "theme": {"type": "string"},
                                "language": {"type": "string"},
                                "timezone": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
        return MelleaTool(
            name="update_profile", tool_call=lambda **k: None, as_json_tool=schema
        )

    def test_nested_object_not_padded(self):
        """(a) Unset nested-object fields must not appear as None in the output."""
        tool = self._nested_object_tool()
        args = {"user_id": "u1", "settings": {"theme": "dark"}}
        validated = validate_tool_arguments(tool, args, strict=False)

        # Exactly what the model sent — no language=None / timezone=None padding.
        assert validated == {"user_id": "u1", "settings": {"theme": "dark"}}
        for unsent in ("language", "timezone"):
            assert unsent not in validated["settings"]

    def test_coercion_still_works(self):
        """(b) Type coercion is unaffected by the no-padding fix."""
        args = {"name": "Test", "age": "30", "score": "95.5", "active": True}
        tool = MelleaTool.from_callable(typed_tool)
        validated = validate_tool_arguments(tool, args, coerce_types=True)

        assert validated["age"] == 30
        assert isinstance(validated["age"], int)
        assert validated["score"] == 95.5
        assert isinstance(validated["score"], float)

    def test_explicit_null_on_nullable_field_preserved(self):
        """(c) A null the model explicitly sent for a nullable field is kept.

        `optional` is typed `str | None`, so an explicit None is a valid, *set*
        value and must survive — only *unset* fields are dropped.
        """
        args = {"required": "value1", "optional": None}
        tool = MelleaTool.from_callable(optional_tool)
        validated = validate_tool_arguments(tool, args, strict=False)

        assert "optional" in validated
        assert validated["optional"] is None

    def test_required_fields_still_present(self):
        """(d) Required fields are still validated and emitted."""
        args = {"required": "value1"}
        tool = MelleaTool.from_callable(optional_tool)
        validated = validate_tool_arguments(tool, args, strict=False)

        assert validated["required"] == "value1"
        # ...and the unset optional field is not padded back in.
        assert "optional" not in validated

    def test_undeclared_extra_key_survives_in_lenient_mode(self):
        """(e) Extra keys the model sent that aren't in the schema survive (lenient)."""
        args = {"message": "hi", "weird_extra": 42}
        tool = MelleaTool.from_callable(simple_tool)
        validated = validate_tool_arguments(tool, args, strict=False)

        assert validated["message"] == "hi"
        assert validated["weird_extra"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
