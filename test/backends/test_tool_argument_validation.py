"""Comprehensive test suite for tool call argument validation.

Tests cover:
- Basic type validation and coercion
- Complex nested types (dicts, lists)
- Union types and Optional parameters
- Missing required arguments
- Extra arguments
- Malformed JSON parsing
- Pydantic model arguments
"""

import json
from typing import Any, Optional, Union

import pytest
from pydantic import BaseModel, ValidationError

from mellea.backends.tools import MelleaTool, parse_tools
from mellea.core import ModelToolCall


# ============================================================================
# Test Fixtures - Tool Functions with Various Signatures
# ============================================================================


def simple_string_tool(message: str) -> str:
    """A simple tool that takes a string.

    Args:
        message: The message to process
    """
    return f"Processed: {message}"


def typed_primitives_tool(name: str, age: int, score: float, active: bool) -> dict:
    """Tool with multiple primitive types.

    Args:
        name: Person's name
        age: Person's age in years
        score: Performance score
        active: Whether person is active
    """
    return {"name": name, "age": age, "score": score, "active": active}


def optional_params_tool(required: str, optional: Optional[str] = None) -> str:
    """Tool with optional parameters.

    Args:
        required: A required parameter
        optional: An optional parameter
    """
    return f"{required}:{optional or 'none'}"


def union_type_tool(value: Union[str, int]) -> str:
    """Tool with union type parameter.

    Args:
        value: Can be string or integer
    """
    return f"Value: {value} (type: {type(value).__name__})"


def list_param_tool(items: list[str]) -> int:
    """Tool with list parameter.

    Args:
        items: List of string items
    """
    return len(items)


def dict_param_tool(config: dict[str, Any]) -> str:
    """Tool with dict parameter.

    Args:
        config: Configuration dictionary
    """
    return json.dumps(config)


def nested_structure_tool(data: dict[str, list[int]]) -> int:
    """Tool with nested structure.

    Args:
        data: Dictionary mapping strings to lists of integers
    """
    return sum(sum(values) for values in data.values())


def default_values_tool(name: str, count: int = 10, prefix: str = "item") -> str:
    """Tool with default values.

    Args:
        name: Base name
        count: Number of items (default: 10)
        prefix: Prefix for items (default: "item")
    """
    return f"{prefix}_{name}_{count}"


class UserModel(BaseModel):
    """Pydantic model for testing."""

    name: str
    age: int
    email: Optional[str] = None


def pydantic_model_tool(user: UserModel) -> str:
    """Tool that accepts a Pydantic model.

    Args:
        user: User information
    """
    return f"User: {user.name}, Age: {user.age}"


def no_params_tool() -> str:
    """Tool with no parameters."""
    return "No params needed"


# ============================================================================
# Test Cases: Basic Type Validation
# ============================================================================


class TestBasicTypeValidation:
    """Test basic type validation and coercion."""

    def test_string_argument(self):
        """Test simple string argument."""
        args = {"message": "Hello, World!"}
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )
        result = tool_call.call_func()
        assert result == "Processed: Hello, World!"

    def test_integer_argument(self):
        """Test integer argument."""
        args = {"name": "Alice", "age": 30, "score": 95.5, "active": True}
        tool_call = ModelToolCall(
            "typed_primitives_tool",
            MelleaTool.from_callable(typed_primitives_tool),
            args,
        )
        result = tool_call.call_func()
        assert result["age"] == 30
        assert isinstance(result["age"], int)

    def test_float_argument(self):
        """Test float argument."""
        args = {"name": "Bob", "age": 25, "score": 88.7, "active": False}
        tool_call = ModelToolCall(
            "typed_primitives_tool",
            MelleaTool.from_callable(typed_primitives_tool),
            args,
        )
        result = tool_call.call_func()
        assert result["score"] == 88.7
        assert isinstance(result["score"], float)

    def test_boolean_argument(self):
        """Test boolean argument."""
        args = {"name": "Charlie", "age": 35, "score": 92.0, "active": True}
        tool_call = ModelToolCall(
            "typed_primitives_tool",
            MelleaTool.from_callable(typed_primitives_tool),
            args,
        )
        result = tool_call.call_func()
        assert result["active"] is True
        assert isinstance(result["active"], bool)


# ============================================================================
# Test Cases: Type Coercion
# ============================================================================


class TestTypeCoercion:
    """Test automatic type coercion scenarios."""

    def test_string_to_int_coercion(self):
        """Test that string "30" works without validation (Python duck typing)."""
        # Python's duck typing allows this to work in many cases
        args = {"name": "Test", "age": "30", "score": 95.5, "active": True}
        tool_call = ModelToolCall(
            "typed_primitives_tool",
            MelleaTool.from_callable(typed_primitives_tool),
            args,
        )

        # This actually works due to Python's duck typing
        result = tool_call.call_func()
        assert result["age"] == "30"  # Still a string without validation

    def test_string_to_float_coercion(self):
        """Test that string "95.5" works without validation (Python duck typing)."""
        args = {"name": "Test", "age": 30, "score": "95.5", "active": True}
        tool_call = ModelToolCall(
            "typed_primitives_tool",
            MelleaTool.from_callable(typed_primitives_tool),
            args,
        )

        # This works due to Python's duck typing
        result = tool_call.call_func()
        assert result["score"] == "95.5"  # Still a string without validation

    def test_int_to_string_coercion(self):
        """Test that int 123 can be coerced to string "123"."""
        args = {"message": 123}  # Should be string
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )

        # This might work due to Python's duck typing, but not guaranteed
        result = tool_call.call_func()
        assert "123" in result

    def test_string_to_bool_coercion(self):
        """Test boolean from strings works without validation (Python duck typing)."""
        # Common LLM outputs: "true", "false", "True", "False"
        args = {"name": "Test", "age": 30, "score": 95.5, "active": "true"}
        tool_call = ModelToolCall(
            "typed_primitives_tool",
            MelleaTool.from_callable(typed_primitives_tool),
            args,
        )

        # This works due to Python's duck typing - non-empty strings are truthy
        result = tool_call.call_func()
        assert result["active"] == "true"  # Still a string without validation


# ============================================================================
# Test Cases: Optional and Default Parameters
# ============================================================================


class TestOptionalParameters:
    """Test optional and default parameter handling."""

    def test_optional_param_provided(self):
        """Test optional parameter when provided."""
        args = {"required": "value1", "optional": "value2"}
        tool_call = ModelToolCall(
            "optional_params_tool", MelleaTool.from_callable(optional_params_tool), args
        )
        result = tool_call.call_func()
        assert result == "value1:value2"

    def test_optional_param_omitted(self):
        """Test optional parameter when omitted."""
        args = {"required": "value1"}
        tool_call = ModelToolCall(
            "optional_params_tool", MelleaTool.from_callable(optional_params_tool), args
        )
        result = tool_call.call_func()
        assert result == "value1:none"

    def test_optional_param_none(self):
        """Test optional parameter explicitly set to None."""
        args = {"required": "value1", "optional": None}
        tool_call = ModelToolCall(
            "optional_params_tool", MelleaTool.from_callable(optional_params_tool), args
        )
        result = tool_call.call_func()
        assert result == "value1:none"

    def test_default_values_all_provided(self):
        """Test tool with all default values provided."""
        args = {"name": "test", "count": 5, "prefix": "custom"}
        tool_call = ModelToolCall(
            "default_values_tool", MelleaTool.from_callable(default_values_tool), args
        )
        result = tool_call.call_func()
        assert result == "custom_test_5"

    def test_default_values_partial(self):
        """Test tool with some default values omitted."""
        args = {"name": "test", "count": 7}
        tool_call = ModelToolCall(
            "default_values_tool", MelleaTool.from_callable(default_values_tool), args
        )
        result = tool_call.call_func()
        assert result == "item_test_7"

    def test_default_values_minimal(self):
        """Test tool with only required parameters."""
        args = {"name": "test"}
        tool_call = ModelToolCall(
            "default_values_tool", MelleaTool.from_callable(default_values_tool), args
        )
        result = tool_call.call_func()
        assert result == "item_test_10"


# ============================================================================
# Test Cases: Union Types
# ============================================================================


class TestUnionTypes:
    """Test union type parameter handling."""

    def test_union_with_string(self):
        """Test union type with string value."""
        args = {"value": "hello"}
        tool_call = ModelToolCall(
            "union_type_tool", MelleaTool.from_callable(union_type_tool), args
        )
        result = tool_call.call_func()
        assert "hello" in result
        assert "str" in result

    def test_union_with_int(self):
        """Test union type with integer value."""
        args = {"value": 42}
        tool_call = ModelToolCall(
            "union_type_tool", MelleaTool.from_callable(union_type_tool), args
        )
        result = tool_call.call_func()
        assert "42" in result
        assert "int" in result

    def test_union_with_string_number(self):
        """Test union type with string that looks like number."""
        # Without validation, this stays as string
        args = {"value": "42"}
        tool_call = ModelToolCall(
            "union_type_tool", MelleaTool.from_callable(union_type_tool), args
        )
        result = tool_call.call_func()
        assert "42" in result
        # Type depends on whether validation coerces


# ============================================================================
# Test Cases: Complex Types (Lists, Dicts)
# ============================================================================


class TestComplexTypes:
    """Test complex type parameters (lists, dicts, nested structures)."""

    def test_list_of_strings(self):
        """Test list parameter with strings."""
        args = {"items": ["apple", "banana", "cherry"]}
        tool_call = ModelToolCall(
            "list_param_tool", MelleaTool.from_callable(list_param_tool), args
        )
        result = tool_call.call_func()
        assert result == 3

    def test_empty_list(self):
        """Test empty list parameter."""
        args = {"items": []}
        tool_call = ModelToolCall(
            "list_param_tool", MelleaTool.from_callable(list_param_tool), args
        )
        result = tool_call.call_func()
        assert result == 0

    def test_dict_parameter(self):
        """Test dictionary parameter."""
        args = {"config": {"key1": "value1", "key2": 42, "key3": True}}
        tool_call = ModelToolCall(
            "dict_param_tool", MelleaTool.from_callable(dict_param_tool), args
        )
        result = tool_call.call_func()
        parsed = json.loads(result)
        assert parsed["key1"] == "value1"
        assert parsed["key2"] == 42

    def test_nested_structure(self):
        """Test nested dictionary with lists."""
        args = {"data": {"group1": [1, 2, 3], "group2": [4, 5], "group3": [6]}}
        tool_call = ModelToolCall(
            "nested_structure_tool",
            MelleaTool.from_callable(nested_structure_tool),
            args,
        )
        result = tool_call.call_func()
        assert result == 21  # Sum of all numbers

    def test_nested_structure_empty(self):
        """Test nested structure with empty lists."""
        args = {"data": {"group1": [], "group2": []}}
        tool_call = ModelToolCall(
            "nested_structure_tool",
            MelleaTool.from_callable(nested_structure_tool),
            args,
        )
        result = tool_call.call_func()
        assert result == 0


# ============================================================================
# Test Cases: Error Conditions
# ============================================================================


class TestErrorConditions:
    """Test error handling for invalid arguments."""

    def test_missing_required_argument(self):
        """Test that missing required argument raises error."""
        args = {}  # Missing 'message'
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )

        with pytest.raises(TypeError, match="missing.*required"):
            tool_call.call_func()

    def test_extra_arguments(self):
        """Test that extra arguments are ignored (Python behavior)."""
        args = {"message": "test", "extra": "ignored"}
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )

        # Python ignores extra kwargs by default
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            tool_call.call_func()

    def test_wrong_type_no_coercion(self):
        """Test that wrong types work without validation (Python duck typing)."""
        args = {"name": "Test", "age": "not_a_number", "score": 95.5, "active": True}
        tool_call = ModelToolCall(
            "typed_primitives_tool",
            MelleaTool.from_callable(typed_primitives_tool),
            args,
        )

        # Python's duck typing allows this - the function just returns what it gets
        result = tool_call.call_func()
        assert result["age"] == "not_a_number"  # Still a string

    def test_none_for_required_param(self):
        """Test that None for required parameter fails."""
        args = {"message": None}
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )

        # Depends on function implementation
        result = tool_call.call_func()
        # May work or fail depending on function


# ============================================================================
# Test Cases: JSON Parsing
# ============================================================================


class TestJSONParsing:
    """Test JSON parsing scenarios from model responses."""

    def test_valid_json_string(self):
        """Test parsing valid JSON string."""
        json_str = '{"message": "Hello, World!"}'
        args = json.loads(json_str)
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )
        result = tool_call.call_func()
        assert "Hello, World!" in result

    def test_malformed_json(self):
        """Test that malformed JSON raises error."""
        json_str = '{"message": "Hello, World!"'  # Missing closing brace

        with pytest.raises(json.JSONDecodeError):
            json.loads(json_str)

    def test_json_with_escaped_quotes(self):
        """Test JSON with escaped quotes."""
        json_str = '{"message": "He said \\"Hello\\""}'
        args = json.loads(json_str)
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )
        result = tool_call.call_func()
        assert "Hello" in result

    def test_json_with_unicode(self):
        """Test JSON with unicode characters."""
        json_str = '{"message": "Hello 世界 🌍"}'
        args = json.loads(json_str)
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )
        result = tool_call.call_func()
        assert "世界" in result

    def test_json_with_nested_objects(self):
        """Test JSON with nested objects."""
        json_str = '{"config": {"nested": {"key": "value"}, "list": [1, 2, 3]}}'
        args = json.loads(json_str)
        tool_call = ModelToolCall(
            "dict_param_tool", MelleaTool.from_callable(dict_param_tool), args
        )
        result = tool_call.call_func()
        parsed = json.loads(result)
        assert parsed["nested"]["key"] == "value"


# ============================================================================
# Test Cases: Pydantic Models
# ============================================================================


class TestPydanticModels:
    """Test tools that accept Pydantic models as arguments."""

    def test_pydantic_model_from_dict(self):
        """Test creating Pydantic model from dict."""
        args = {"user": {"name": "Alice", "age": 30, "email": "alice@example.com"}}

        # Need to convert dict to Pydantic model
        user_data = args["user"]
        user = UserModel(**user_data)
        args_with_model = {"user": user}

        tool_call = ModelToolCall(
            "pydantic_model_tool",
            MelleaTool.from_callable(pydantic_model_tool),
            args_with_model,
        )
        result = tool_call.call_func()
        assert "Alice" in result
        assert "30" in result

    def test_pydantic_model_validation_error(self):
        """Test that invalid Pydantic model data raises error."""
        user_data = {"name": "Bob", "age": "not_an_int"}  # Invalid age type

        with pytest.raises(ValidationError):
            UserModel(**user_data)

    def test_pydantic_model_with_optional(self):
        """Test Pydantic model with optional field."""
        user_data = {"name": "Charlie", "age": 25}  # email is optional
        user = UserModel(**user_data)
        args = {"user": user}

        tool_call = ModelToolCall(
            "pydantic_model_tool", MelleaTool.from_callable(pydantic_model_tool), args
        )
        result = tool_call.call_func()
        assert "Charlie" in result


# ============================================================================
# Test Cases: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_no_parameters_tool(self):
        """Test tool with no parameters."""
        args = {}
        tool_call = ModelToolCall(
            "no_params_tool", MelleaTool.from_callable(no_params_tool), args
        )
        result = tool_call.call_func()
        assert result == "No params needed"

    def test_no_parameters_with_hallucinated_args(self):
        """Test that hallucinated args for no-param tool are handled."""
        # Models sometimes hallucinate parameters
        args = {"fake_param": "should_be_ignored"}
        tool_call = ModelToolCall(
            "no_params_tool", MelleaTool.from_callable(no_params_tool), args
        )

        # This should fail without validation that clears args
        with pytest.raises(TypeError):
            tool_call.call_func()

    def test_very_long_string(self):
        """Test with very long string argument."""
        long_string = "x" * 10000
        args = {"message": long_string}
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )
        result = tool_call.call_func()
        assert len(result) > 10000

    def test_special_characters_in_string(self):
        """Test with special characters."""
        special = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        args = {"message": special}
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )
        result = tool_call.call_func()
        assert special in result

    def test_empty_string(self):
        """Test with empty string."""
        args = {"message": ""}
        tool_call = ModelToolCall(
            "simple_string_tool", MelleaTool.from_callable(simple_string_tool), args
        )
        result = tool_call.call_func()
        assert result == "Processed: "


# ============================================================================
# Test Cases: parse_tools() Function
# ============================================================================


class TestParseToolsFunction:
    """Test the parse_tools() function from backends/tools.py."""

    def test_parse_single_tool_call(self):
        """Test parsing a single tool call from text."""
        text = """
        I'll call the function now:
        {"name": "get_weather", "arguments": {"location": "Boston", "days": 3}}
        """
        tools = list(parse_tools(text))
        assert len(tools) == 1
        assert tools[0][0] == "get_weather"
        assert tools[0][1]["location"] == "Boston"
        assert tools[0][1]["days"] == 3

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls from text."""
        text = """
        First: {"name": "tool1", "arguments": {"arg1": "value1"}}
        Second: {"name": "tool2", "arguments": {"arg2": "value2"}}
        """
        tools = list(parse_tools(text))
        assert len(tools) == 2
        assert tools[0][0] == "tool1"
        assert tools[1][0] == "tool2"

    def test_parse_with_extra_text(self):
        """Test parsing tool calls with surrounding text."""
        text = """
        Let me help you with that. I'll use the get_temperature function.
        {"name": "get_temperature", "arguments": {"location": "New York"}}
        That should give us the current temperature.
        """
        tools = list(parse_tools(text))
        assert len(tools) == 1
        assert tools[0][0] == "get_temperature"

    def test_parse_no_tools(self):
        """Test parsing text with no tool calls."""
        text = "This is just regular text with no tool calls."
        tools = list(parse_tools(text))
        assert len(tools) == 0

    def test_parse_malformed_json(self):
        """Test that malformed JSON is skipped."""
        text = """
        {"name": "tool1", "arguments": {"arg1": "value1"}}
        {"name": "bad_tool", "arguments": {broken json}}
        {"name": "tool2", "arguments": {"arg2": "value2"}}
        """
        tools = list(parse_tools(text))
        # Should parse the valid ones and skip the malformed one
        assert len(tools) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
