# pytest: ollama, llm
"""Example demonstrating the @tool decorator for cleaner tool definitions."""

import ast

from mellea import start_session
from mellea.backends import ModelOption, tool


# Define tools using the @tool decorator - much cleaner than MelleaTool.from_callable()
@tool
def get_weather(location: str, days: int = 1) -> dict:
    """Get weather forecast for a location.

    Args:
        location: City name
        days: Number of days to forecast (default: 1)
    """
    # Mock implementation
    return {"location": location, "days": days, "forecast": "sunny", "temperature": 72}


@tool
def search_web(query: str, max_results: int = 5) -> list[str]:
    """Search the web for information.

    Args:
        query: Search query
        max_results: Maximum number of results to return
    """
    # Mock implementation
    return [f"Result {i + 1} for '{query}'" for i in range(max_results)]


@tool(name="calculator")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate
    """
    try:
        # Use ast.literal_eval for safe evaluation of simple expressions
        result = ast.literal_eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e!s}"


def example_basic_usage():
    """Example 1: Basic usage with decorated tools."""
    print("\n=== Example 1: Basic Tool Usage ===")

    # Without the decorator, you can add tools using:
    # tools = [MelleaTool.from_callable(get_weather), MelleaTool.from_callable(search_web)]

    # Now you can just pass the decorated functions directly to model_options
    # Example: model_options={ModelOption.TOOLS: [get_weather, search_web, calculate]}

    # The decorated functions still work as normal functions
    weather = get_weather("Boston", days=3)
    print(f"Direct call: {weather}")

    # And they have tool properties
    print(f"Tool name: {get_weather.name}")
    print(f"Tool has JSON schema: {'function' in get_weather.as_json_tool}")


def example_with_llm():
    """Example 2: Using decorated tools with an LLM."""
    print("\n=== Example 2: Using Tools with LLM ===")

    m = start_session()

    # Pass decorated tools directly - no wrapping needed!
    response = m.instruct(
        description="What's the weather like in San Francisco?",
        model_options={ModelOption.TOOLS: [get_weather, search_web]},
    )

    print(f"Response: {response}")


def example_custom_name():
    """Example 3: Using custom tool names."""
    print("\n=== Example 3: Custom Tool Names ===")

    # The calculator tool was decorated with @tool(name="calculator")
    # So its name is "calculator" instead of "calculate"
    print("Function name: calculate")
    print(f"Tool name: {calculate.name}")

    # Both work the same way
    result = calculate("2 + 2")
    print(f"Result: {result}")


def example_comparison():
    """Example 4: Comparison of old vs new approach."""
    print("\n=== Example 4: Old vs New Approach ===")

    # OLD APPROACH (still works, but verbose):
    from mellea.backends.tools import MelleaTool

    def old_style_tool(x: int) -> int:
        """Old style tool.

        Args:
            x: Input value
        """
        return x * 2

    old_tool = MelleaTool.from_callable(old_style_tool)
    print(f"Old approach - tool name: {old_tool.name}")

    # NEW APPROACH (cleaner):
    @tool
    def new_style_tool(x: int) -> int:
        """New style tool.

        Args:
            x: Input value
        """
        return x * 2

    print(f"New approach - tool name: {new_style_tool.name}")

    # Both can be used together in a tools list
    tools = [old_tool, new_style_tool, get_weather]
    print(f"Mixed tools list: {[t.name for t in tools]}")


if __name__ == "__main__":
    example_basic_usage()
    # example_with_llm()  # Uncomment to test with actual LLM
    example_custom_name()
    example_comparison()
