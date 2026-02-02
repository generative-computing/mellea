"""Integration tests for the validate_tool_arguments function.

These tests verify that the validation function works correctly with
the actual tool call flow.
"""

import pytest
from typing import Any, Optional, Union

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


def optional_tool(required: str, optional: Optional[str] = None) -> str:
    """Tool with optional parameters.

    Args:
        required: A required parameter
        optional: An optional parameter
    """
    return f"{required}:{optional or 'none'}"


def union_tool(value: Union[str, int]) -> str:
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
        assert "optional" not in validated or validated.get("optional") is None

    def test_optional_param_none(self):
        """Test validation when optional parameter is explicitly None."""
        args = {"required": "value1", "optional": None}
        tool = MelleaTool.from_callable(optional_tool)
        validated = validate_tool_arguments(tool, args)

        assert validated["required"] == "value1"
        assert validated["optional"] is None


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


class TestEdgeCases:
    """Test edge cases."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
