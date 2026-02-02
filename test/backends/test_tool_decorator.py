"""Tests for the @tool decorator."""

import pytest

from mellea.backends import MelleaTool, tool
from mellea.core import ModelToolCall


# ============================================================================
# Test Fixtures - Tool Functions
# ============================================================================


@tool
def simple_tool(message: str) -> str:
    """A simple tool that takes a string.

    Args:
        message: The message to process
    """
    return f"Processed: {message}"


@tool(name="custom_name")
def tool_with_custom_name(value: int) -> int:
    """Tool with custom name.

    Args:
        value: A value to process
    """
    return value * 2


@tool
def multi_param_tool(name: str, age: int, active: bool = True) -> dict:
    """Tool with multiple parameters.

    Args:
        name: Person's name
        age: Person's age
        active: Whether active
    """
    return {"name": name, "age": age, "active": active}


def undecorated_function(x: int) -> int:
    """A regular function without the decorator.

    Args:
        x: Input value
    """
    return x + 1


# ============================================================================
# Test Cases: Basic Decorator Functionality
# ============================================================================


class TestToolDecoratorBasics:
    """Test basic decorator functionality."""

    def test_decorated_function_is_callable(self):
        """Test that decorated function can still be called normally."""
        result = simple_tool("hello")
        assert result == "Processed: hello"

    def test_decorated_function_has_name_attribute(self):
        """Test that decorated function has name attribute."""
        assert hasattr(simple_tool, "name")
        assert simple_tool.name == "simple_tool"

    def test_decorated_function_has_as_json_tool(self):
        """Test that decorated function has as_json_tool property."""
        assert hasattr(simple_tool, "as_json_tool")
        json_tool = simple_tool.as_json_tool
        assert isinstance(json_tool, dict)
        assert "function" in json_tool

    def test_decorated_function_has_run_method(self):
        """Test that decorated function has run method."""
        assert hasattr(simple_tool, "run")
        result = simple_tool.run("test")
        assert result == "Processed: test"

    def test_decorated_function_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        assert simple_tool.__name__ == "simple_tool"
        assert "simple tool" in simple_tool.__doc__.lower()

    def test_custom_name_decorator(self):
        """Test decorator with custom name parameter."""
        assert tool_with_custom_name.name == "custom_name"
        # Function should still work
        result = tool_with_custom_name(5)
        assert result == 10


# ============================================================================
# Test Cases: Integration with MelleaTool
# ============================================================================


class TestToolDecoratorIntegration:
    """Test integration with existing MelleaTool infrastructure."""

    def test_decorated_tool_in_list(self):
        """Test that decorated tools can be used in a list."""
        tools = [simple_tool, multi_param_tool]
        assert len(tools) == 2
        # Should be able to access tool properties
        assert tools[0].name == "simple_tool"
        assert tools[1].name == "multi_param_tool"

    def test_decorated_tool_with_model_tool_call(self):
        """Test that decorated tools work with ModelToolCall."""
        args = {"message": "test message"}
        # Decorated function IS a MelleaTool, can be passed directly
        tool_call = ModelToolCall("simple_tool", simple_tool, args)
        result = tool_call.call_func()
        assert result == "Processed: test message"

    def test_decorated_tool_json_schema(self):
        """Test that decorated tool generates correct JSON schema."""
        json_tool = simple_tool.as_json_tool
        assert json_tool["type"] == "function"
        assert json_tool["function"]["name"] == "simple_tool"
        assert "parameters" in json_tool["function"]
        properties = json_tool["function"]["parameters"]["properties"]
        assert "message" in properties
        assert properties["message"]["type"] == "string"

    def test_multi_param_tool_schema(self):
        """Test schema generation for multi-parameter tool."""
        json_tool = multi_param_tool.as_json_tool
        properties = json_tool["function"]["parameters"]["properties"]
        assert "name" in properties
        assert "age" in properties
        assert "active" in properties
        # Check required fields
        required = json_tool["function"]["parameters"]["required"]
        assert "name" in required
        assert "age" in required
        # active has default, so might not be required


# ============================================================================
# Test Cases: Comparison with from_callable
# ============================================================================


class TestToolDecoratorVsFromCallable:
    """Test that decorator produces equivalent results to from_callable."""

    def test_decorator_equivalent_to_from_callable(self):
        """Test that @tool produces same result as MelleaTool.from_callable."""
        # Create tool using from_callable
        manual_tool = MelleaTool.from_callable(undecorated_function)

        # Create tool using decorator
        @tool
        def decorated_version(x: int) -> int:
            """A regular function without the decorator.

            Args:
                x: Input value
            """
            return x + 1

        # Compare JSON schemas
        manual_json = manual_tool.as_json_tool
        decorated_json = decorated_version.as_json_tool

        # Names should match
        assert manual_json["function"]["name"] == "undecorated_function"
        assert decorated_json["function"]["name"] == "decorated_version"

        # Parameters should have same structure
        assert (
            manual_json["function"]["parameters"]["type"]
            == decorated_json["function"]["parameters"]["type"]
        )

    def test_both_approaches_work_in_tools_list(self):
        """Test that both decorated and from_callable tools work together."""
        manual_tool = MelleaTool.from_callable(undecorated_function)
        tools = [simple_tool, manual_tool]

        # Both should have name attribute
        assert hasattr(tools[0], "name")
        assert hasattr(tools[1], "name")

        # Both should have as_json_tool
        assert hasattr(tools[0], "as_json_tool")
        assert hasattr(tools[1], "as_json_tool")


# ============================================================================
# Test Cases: Edge Cases
# ============================================================================


class TestToolDecoratorEdgeCases:
    """Test edge cases and error conditions."""

    def test_decorator_with_no_params_function(self):
        """Test decorator on function with no parameters."""

        @tool
        def no_params() -> str:
            """Function with no parameters."""
            return "no params"

        result = no_params()
        assert result == "no params"
        assert no_params.name == "no_params"

    def test_decorator_preserves_function_behavior(self):
        """Test that decorator doesn't change function behavior."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: First number
                b: Second number
            """
            return a + b

        # Should work exactly like the original function
        assert add(2, 3) == 5
        assert add(10, 20) == 30
        assert add.run(5, 7) == 12

    def test_decorator_with_complex_types(self):
        """Test decorator with complex parameter types."""

        @tool
        def complex_tool(items: list[str], config: dict) -> int:
            """Tool with complex types.

            Args:
                items: List of items
                config: Configuration dict
            """
            return len(items) + len(config)

        result = complex_tool(["a", "b"], {"x": 1, "y": 2})
        assert result == 4

    def test_multiple_decorators_on_same_function(self):
        """Test that decorator can be applied multiple times (creates new instances)."""

        def base_func(x: int) -> int:
            """Base function.

            Args:
                x: Input
            """
            return x

        tool1 = tool(base_func)
        tool2 = tool(name="custom")(base_func)

        assert tool1.name == "base_func"
        assert tool2.name == "custom"


# ============================================================================
# Test Cases: Usage Patterns
# ============================================================================


class TestToolDecoratorUsagePatterns:
    """Test common usage patterns."""

    def test_tools_in_dict(self):
        """Test using decorated tools in a dictionary."""
        tools_dict = {"simple": simple_tool, "multi": multi_param_tool}

        assert tools_dict["simple"].name == "simple_tool"
        assert tools_dict["multi"].name == "multi_param_tool"

    def test_tools_passed_to_function(self):
        """Test passing decorated tools to a function."""

        def process_tools(tool_list):
            """Process a list of tools."""
            return [t.name for t in tool_list]

        tools = [simple_tool, multi_param_tool]
        names = process_tools(tools)
        assert "simple_tool" in names
        assert "multi_param_tool" in names

    def test_accessing_underlying_mellea_tool(self):
        """Test that decorated function IS a MelleaTool instance."""
        assert isinstance(simple_tool, MelleaTool)
        assert simple_tool.name == "simple_tool"
        # Verify it has all MelleaTool properties
        assert hasattr(simple_tool, "as_json_tool")
        assert hasattr(simple_tool, "run")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
