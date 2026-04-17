# pytest: ollama, e2e

"""Example demonstrating tool calling with m serve.

This example shows how to use the OpenAI-compatible tool calling API
with m serve. The server will accept tool definitions and return tool
calls in the response when the model decides to use them.
"""

from typing import Any

import mellea
from cli.serve.models import ChatMessage
from mellea.backends import ModelOption
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


class GetStockPriceTool(AbstractMelleaTool):
    """Tool for getting stock price information."""

    name = "get_stock_price"

    def run(self, symbol: str) -> str:
        """Get the current stock price for a symbol.

        Args:
            symbol: The stock ticker symbol (e.g., AAPL, GOOGL)

        Returns:
            Stock price information as a string
        """
        # In a real implementation, this would call a stock market API
        mock_prices = {
            "AAPL": "$175.43",
            "GOOGL": "$142.87",
            "MSFT": "$378.91",
            "TSLA": "$242.15",
        }
        price = mock_prices.get(symbol.upper(), "$100.00")
        return f"The current price of {symbol.upper()} is {price}"

    @property
    def as_json_tool(self) -> dict[str, Any]:
        """Return JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get the current stock price for a given ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock ticker symbol, e.g. AAPL, GOOGL",
                        }
                    },
                    "required": ["symbol"],
                },
            },
        }


# Create tool instances
weather_tool = GetWeatherTool()
stock_price_tool = GetStockPriceTool()

# Map tool names to instances for easy lookup
TOOLS = {weather_tool.name: weather_tool, stock_price_tool.name: stock_price_tool}


def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: None | dict = None,
) -> ModelOutputThunk:
    """Serve function that handles tool calling.

    This function demonstrates how to use tools with m serve. The tools
    are passed via model_options using ModelOption.TOOLS, and tool_choice
    can be specified using ModelOption.TOOL_CHOICE.

    Args:
        input: List of chat messages
        requirements: Optional list of requirement strings
        model_options: Model options including ModelOption.TOOLS and ModelOption.TOOL_CHOICE

    Returns:
        ModelOutputThunk with potential tool calls
    """
    requirements = requirements if requirements else []
    message = input[-1].content

    # Extract tools from model_options if provided
    tools = None
    if model_options and ModelOption.TOOLS in model_options:
        # Convert OpenAI tool format to Mellea tool format
        openai_tools = model_options[ModelOption.TOOLS]
        tools = {}
        for tool_def in openai_tools:
            tool_name = tool_def["function"]["name"]
            if tool_name in TOOLS:
                tools[tool_name] = TOOLS[tool_name]

    # Build model options with tools
    final_model_options = model_options or {}
    if tools:
        final_model_options[ModelOption.TOOLS] = tools

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

    # Simulate tool definitions being passed with tool_choice
    test_model_options = {
        ModelOption.TOOLS: [weather_tool.as_json_tool, stock_price_tool.as_json_tool],
        ModelOption.TOOL_CHOICE: "auto",  # Can be "none", "auto", or specific tool
    }

    response = serve(input=test_messages, model_options=test_model_options)

    print(f"Response: {response.value}")
    if response.tool_calls:
        print(f"Tool calls requested: {list(response.tool_calls.keys())}")
