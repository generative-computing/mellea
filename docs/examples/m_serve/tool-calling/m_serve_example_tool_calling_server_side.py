# pytest: ollama, e2e

"""m serve module that executes tool calls server-side before responding.

Unlike the standard tool-calling example (which surfaces ``function_call``
output items for the client to handle), this module runs the agentic loop
entirely on the server:

1. ``serve()`` calls the model with the tool definitions.
2. If the model requests tool calls, the server executes them using
   ``call_tools()`` and appends the results to the session context.
3. The model is called again with the tool results in context.
4. Steps 2-3 repeat until the model stops requesting tools (up to
   ``MAX_TOOL_ROUNDS`` rounds).
5. A single, fully-resolved response is returned to the client.

The client therefore never sees intermediate ``function_call`` items — it
only receives the final answer.

Usage:
    1. Start the server:
       uv run m serve docs/examples/m_serve/tool-calling/m_serve_example_tool_calling_server_side.py

    2. Run the matching client:
       uv run python docs/examples/m_serve/tool-calling/client_responses_tool_calling_server_side.py
"""

import os
from typing import Any

from mellea.backends import ModelOption
from mellea.backends.model_ids import IBM_GRANITE_4_HYBRID_MICRO
from mellea.backends.openai import OpenAIBackend
from mellea.backends.tools import MelleaTool
from mellea.core import ModelOutputThunk, Requirement
from mellea.core.base import AbstractMelleaTool
from mellea.formatters import TemplateFormatter
from mellea.serve import ChatMessage
from mellea.stdlib.components.chat import Message
from mellea.stdlib.functional import call_tools
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

# Maximum number of tool-call → result rounds before giving up.
MAX_TOOL_ROUNDS = 5


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
        resolved_units = units or "celsius"
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


weather_tool_impl = GetWeatherTool()
stock_price_tool_impl = GetStockPriceTool()

# MelleaTool wrappers used by the direct __main__ path.
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

# Lookup map used by serve() to resolve OpenAI-style tool defs to implementations.
TOOLS: dict[str, AbstractMelleaTool] = {
    weather_tool_impl.name: weather_tool_impl,
    stock_price_tool_impl.name: stock_price_tool_impl,
}


def _resolve_tools(model_options: dict | None) -> dict[str, AbstractMelleaTool]:
    """Map tool definitions from model_options to server-side implementations.

    Accepts both OpenAI-style JSON dicts (from the /v1/responses path) and
    native ``AbstractMelleaTool`` instances (from the ``__main__`` path).
    """
    if model_options is None or ModelOption.TOOLS not in model_options:
        return {}

    result: dict[str, AbstractMelleaTool] = {}
    for tool_def in model_options[ModelOption.TOOLS]:
        if isinstance(tool_def, AbstractMelleaTool):
            result[tool_def.name] = tool_def
        else:
            tool_name = tool_def["function"]["name"]
            if tool_name in TOOLS:
                result[tool_name] = TOOLS[tool_name]
    return result


def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: None | dict = None,
) -> ModelOutputThunk:
    """Serve function that executes tool calls server-side.

    The agentic loop runs entirely within this function:

    1. Call the model with the user message and tool definitions.
    2. If the model requests tool calls, execute them with ``call_tools()``
       and add the results to the session context.
    3. Call the model again to synthesise a final answer from the results.
    4. Repeat until the model stops requesting tools or ``MAX_TOOL_ROUNDS``
       is reached.

    The caller (client) receives a fully-resolved ``ModelOutputThunk`` — no
    ``function_call`` items are surfaced to the client.

    Args:
        input: List of chat messages from the request.
        requirements: Optional list of requirement strings (unused here).
        model_options: Model options forwarded from the request, including
            ``ModelOption.TOOLS`` and ``ModelOption.TOOL_CHOICE``.

    Returns:
        A ``ModelOutputThunk`` containing the final text response.
    """
    tools = _resolve_tools(model_options)
    final_model_options = dict(model_options or {})

    # Narrow advertised tools to just the one requested via tool_choice, if any.
    if tools and model_options is not None:
        tool_choice = model_options.get(ModelOption.TOOL_CHOICE)
        if isinstance(tool_choice, dict):
            selected_name = tool_choice.get("function", {}).get("name")
            if selected_name and selected_name in tools:
                tools = {selected_name: tools[selected_name]}
    if tools:
        final_model_options[ModelOption.TOOLS] = tools

    message = input[-1].content
    session = MelleaSession(backend)

    # First call — model may emit tool_calls instead of (or alongside) text.
    result = session.instruct(
        description=message,  # type: ignore[arg-type]
        requirements=[Requirement(r) for r in (requirements or [])],  # type: ignore[arg-type]
        model_options=final_model_options,
        tool_calls=bool(tools),
        strategy=None,
    )

    # Agentic loop: execute requested tools and re-prompt until the model
    # produces a plain text answer (no more tool calls).
    for _ in range(MAX_TOOL_ROUNDS):
        if not result.tool_calls:
            break

        # Execute every requested tool call and add results to the context.
        tool_messages = call_tools(result, backend)
        for tool_msg in tool_messages:
            session.ctx = session.ctx.add(tool_msg)

        # Re-prompt the model; it now has the tool results in context and
        # should produce a final answer rather than another tool call.
        result = session.instruct(
            description="Answer the user's question using the tool results above.",
            model_options={
                ModelOption.MAX_NEW_TOKENS: final_model_options.get(
                    ModelOption.MAX_NEW_TOKENS, 1000
                )
            },
            tool_calls=False,
            strategy=None,
        )

    return result


if __name__ == "__main__":
    # Quick smoke test: the whole tool loop runs locally.
    session = MelleaSession(backend)
    result = session.instruct(
        "What's the weather in Boston?",
        model_options={
            ModelOption.TOOLS: [weather_tool],
            ModelOption.TOOL_CHOICE: "auto",
            ModelOption.MAX_NEW_TOKENS: 1000,
        },
        strategy=None,
        tool_calls=True,
    )

    if result.tool_calls:
        tool_messages = call_tools(result, backend)
        for tool_msg in tool_messages:
            session.ctx = session.ctx.add(tool_msg)
            print(f"Tool executed: {tool_msg.name} → {tool_msg._tool_output}")

        result = session.instruct(
            "Answer the user's question using the tool results above.",
            model_options={ModelOption.MAX_NEW_TOKENS: 1000},
            tool_calls=False,
            strategy=None,
        )

    print(f"Final answer: {result.value}")
