# pytest: ollama, e2e

"""Example demonstrating tool calling with m serve.

This example shows how to use the OpenAI-compatible tool calling API
with m serve. The server will accept tool definitions and return tool
calls in the response when the model decides to use them.
"""

from typing import Any

import mellea
from cli.serve.models import ChatMessage
from mellea.core import ModelOutputThunk, Requirement
from mellea.core.base import AbstractMelleaTool
from mellea.stdlib.context import ChatContext

session = mellea.start_session(ctx=ChatContext())


class GetWeatherTool(AbstractMelleaTool):
    """Tool for getting weather information."""

    name = "get_weather"

    def run(self, location: str, units: str = "celsius") -> str:
        """Get the current weather for a location.

        Args:
            location: The city name
            units: Temperature units (celsius or fahrenheit)

        Returns:
            Weather information as a string
        """
        # In a real implementation, this would call a weather API
        return f"The weather in {location} is sunny and 22°{units[0].upper()}"

    @property
    def as_json_tool(self) -> dict[str, Any]:
        """Return JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units",
                        },
                    },
                    "required": ["location"],
                },
            },
        }


class CalculatorTool(AbstractMelleaTool):
    """Tool for performing calculations."""

    name = "calculator"

    def run(self, expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: A mathematical expression to evaluate

        Returns:
            The result of the calculation
        """
        try:
            # In a real implementation, use a safe expression evaluator
            result = eval(expression)  # noqa: S307
            return f"The result is {result}"
        except Exception as e:
            return f"Error evaluating expression: {e}"

    @property
    def as_json_tool(self) -> dict[str, Any]:
        """Return JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate, e.g. '2 + 2'",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }


# Create tool instances
weather_tool = GetWeatherTool()
calculator_tool = CalculatorTool()

# Map tool names to instances for easy lookup
TOOLS = {weather_tool.name: weather_tool, calculator_tool.name: calculator_tool}


def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: None | dict = None,
) -> ModelOutputThunk:
    """Serve function that handles tool calling.

    This function demonstrates how to use tools with m serve. The tools
    are passed via model_options and the model can request to call them.

    Args:
        input: List of chat messages
        requirements: Optional list of requirement strings
        model_options: Model options including tools and tool_choice

    Returns:
        ModelOutputThunk with potential tool calls
    """
    requirements = requirements if requirements else []
    message = input[-1].content

    # Extract tools from model_options if provided
    tools = None
    if model_options and "@@@tools@@@" in model_options:
        # Convert OpenAI tool format to Mellea tool format
        openai_tools = model_options["@@@tools@@@"]
        tools = {}
        for tool_def in openai_tools:
            tool_name = tool_def["function"]["name"]
            if tool_name in TOOLS:
                tools[tool_name] = TOOLS[tool_name]

    # Build model options with tools
    final_model_options = model_options or {}
    if tools:
        final_model_options["@@@tools@@@"] = tools

    # Use instruct to generate response with potential tool calls
    result = session.instruct(
        description=message,  # type: ignore
        requirements=[Requirement(req) for req in requirements],  # type: ignore
        model_options=final_model_options,
    )

    return result


if __name__ == "__main__":
    # Example usage (for testing purposes)
    test_messages = [ChatMessage(role="user", content="What's the weather in Paris?")]

    # Simulate tool definitions being passed
    test_model_options = {
        "@@@tools@@@": [weather_tool.as_json_tool, calculator_tool.as_json_tool]
    }

    response = serve(input=test_messages, model_options=test_model_options)

    print(f"Response: {response.value}")
    if response.tool_calls:
        print(f"Tool calls requested: {list(response.tool_calls.keys())}")
