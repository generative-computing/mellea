# pytest: ollama, e2e

"""Example demonstrating tool calling with m serve.

This file supports two distinct usage patterns:

1. Running it directly with ``uv run python ...`` performs a local smoke test
   using native Mellea tool calling.
2. Serving it with ``m serve`` exposes an OpenAI-compatible endpoint that
   accepts OpenAI-style tool definitions in the request.

The direct ``__main__`` smoke test is intentionally separate from the
OpenAI-compatible server flow because local ``session.instruct(...)`` calls
should use ``MelleaTool`` objects directly.
"""

import os
from typing import Any

from cli.serve.models import ChatMessage
from mellea.backends import ModelOption
from mellea.backends.model_ids import IBM_GRANITE_4_HYBRID_MICRO
from mellea.backends.openai import OpenAIBackend
from mellea.backends.tools import MelleaTool
from mellea.core import ModelOutputThunk, Requirement
from mellea.core.base import AbstractMelleaTool
from mellea.formatters import TemplateFormatter
from mellea.stdlib.context import ChatContext
from mellea.stdlib.session import MelleaSession

_ollama_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
if not _ollama_host.startswith(("http://", "https://")):
    _ollama_host = f"http://{_ollama_host}"

backend = OpenAIBackend(
    model_id=IBM_GRANITE_4_HYBRID_MICRO.ollama_name,  # type: ignore[arg-type]
    formatter=TemplateFormatter(model_id=IBM_GRANITE_4_HYBRID_MICRO.hf_model_name),  # type: ignore[arg-type]
    base_url=f"{_ollama_host}/v1",
    api_key="ollama",
)
session = MelleaSession(backend, ctx=ChatContext())


class GetWeatherTool(AbstractMelleaTool):
    """Tool for getting weather information."""

    name = "get_weather"

    def run(self, location: str, units: str | None = "celsius") -> str:
        """Get the current weather for a location.

        Args:
            location: The city name
            units: Temperature units (celsius or fahrenheit)

        Returns:
            Weather information as a string
        """
        # Models sometimes emit optional arguments explicitly as null/None.
        resolved_units = units or "celsius"
        # In a real implementation, this would call a weather API
        return f"The weather in {location} is sunny and 22°{resolved_units[0].upper()}"

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


# Create tool instances for server-side lookup
weather_tool_impl = GetWeatherTool()
stock_price_tool_impl = GetStockPriceTool()

# Native MelleaTool wrappers are only needed for the direct ``__main__`` path.
# The backend helper used by local ``session.instruct(..., ModelOption.TOOLS=[...])``
# expects ``MelleaTool`` instances in a list, while the server path below uses the
# class-based implementations via the ``TOOLS`` lookup.
weather_tool = MelleaTool(
    name=weather_tool_impl.name,
    tool_call=weather_tool_impl.run,
    as_json_tool=weather_tool_impl.as_json_tool,
)
stock_price_tool = MelleaTool(
    name=stock_price_tool_impl.name,
    tool_call=stock_price_tool_impl.run,
    as_json_tool=stock_price_tool_impl.as_json_tool,
)

# Map tool names to server-side tool implementations for easy lookup
TOOLS = {
    weather_tool_impl.name: weather_tool_impl,
    stock_price_tool_impl.name: stock_price_tool_impl,
}


def _extract_mellea_tools_from_model_options(
    model_options: dict | None,
) -> dict[str, AbstractMelleaTool]:
    """Normalize example tool inputs to native tool instances.

    This example supports only two shapes:
    - OpenAI-style JSON tool definitions from the server path
    - native tool objects from the direct ``__main__`` path
    """
    if model_options is None or ModelOption.TOOLS not in model_options:
        return {}

    provided_tools = model_options[ModelOption.TOOLS]
    tools: dict[str, AbstractMelleaTool] = {}

    for tool_def in provided_tools:
        if isinstance(tool_def, AbstractMelleaTool):
            tools[tool_def.name] = tool_def
        else:
            tool_name = tool_def["function"]["name"]
            if tool_name in TOOLS:
                tools[tool_name] = TOOLS[tool_name]

    return tools


def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: None | dict = None,
) -> ModelOutputThunk:
    """Serve function that handles tool calling.

    This function demonstrates how to use tools with m serve. The tools
    are passed via model_options using ModelOption.TOOLS, and tool_choice
    can be specified using ModelOption.TOOL_CHOICE. Mellea forwards that
    setting to compatible backends, but the downstream provider/model may
    still ignore it or treat it as a weak preference.

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
    tools = _extract_mellea_tools_from_model_options(model_options)

    # Build model options with tools.
    # If the caller explicitly selected a single function via tool_choice,
    # narrow the advertised tool set to that one tool so the backend/model
    # is not asked to choose among unrelated tools.
    final_model_options = dict(model_options or {})
    selected_tool_name: str | None = None
    if tools:
        selected_tools = tools
        if model_options is not None and ModelOption.TOOL_CHOICE in model_options:
            tool_choice = model_options[ModelOption.TOOL_CHOICE]
            if isinstance(tool_choice, dict):
                selected_tool_name = tool_choice.get("function", {}).get("name")
                if selected_tool_name in tools:
                    selected_tools = {selected_tool_name: tools[selected_tool_name]}
        final_model_options[ModelOption.TOOLS] = selected_tools

    # Keep the serve path deterministic for the client example by retrying only
    # at the request level. Enforcing uses_tool(...) inside session.instruct()
    # caused noisy server-side failures when the model ignored the tool request
    # on a particular sample.
    result = session.instruct(
        description=message,  # type: ignore
        requirements=[Requirement(req) for req in requirements],  # type: ignore
        model_options=final_model_options,
        tool_calls=True,
        strategy=None,
    )

    return result


if __name__ == "__main__":
    response = session.instruct(
        "What's the weather in Boston?",
        model_options={
            ModelOption.TOOLS: [weather_tool],
            # This direct path now uses the OpenAI backend against Ollama's
            # OpenAI-compatible endpoint, so TOOL_CHOICE is forwarded by
            # Mellea. Ollama and/or the selected model may still ignore it
            # or not enforce it strictly in practice.
            ModelOption.TOOL_CHOICE: "auto",
            ModelOption.MAX_NEW_TOKENS: 1000,
        },
        strategy=None,
        tool_calls=True,
    )

    print(f"Response: {response.value}")
    print(
        "Tool calls requested:",
        None if response.tool_calls is None else list(response.tool_calls.keys()),
    )

    if response.tool_calls and weather_tool.name in response.tool_calls:
        tool_result = response.tool_calls[weather_tool.name].call_func()
        print(f"Tool result: {tool_result}")
