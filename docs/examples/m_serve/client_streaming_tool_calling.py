"""Client example for testing streaming with tool calling.

This script demonstrates how to use streaming responses with tool calls
from an m serve server using the OpenAI-compatible API.

Usage:
    1. Start the server:
       uv run m serve docs/examples/m_serve/m_serve_example_tool_calling.py

    2. Run this client:
       uv run python docs/examples/m_serve/client_streaming_tool_calling.py
"""

import json
from typing import Any

import requests

# Server configuration
BASE_URL = "http://localhost:8080"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"

# Define tools in OpenAI format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
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
    },
]


def make_streaming_request(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_name: str | None = None,
) -> tuple[str, list[dict[str, Any]] | None, str]:
    """Make a streaming request to the m serve API.

    Args:
        messages: List of message dictionaries
        tools: Optional list of tool definitions
        tool_name: Optional tool name to request explicitly

    Returns:
        Tuple of (content, tool_calls, finish_reason)
    """
    payload: dict[str, Any] = {
        "model": "gpt-3.5-turbo",  # Model name (not used by m serve)
        "messages": messages,
        "temperature": 0.7,
        "stream": True,
    }

    if tools:
        payload["tools"] = tools
        if tool_name is not None:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_name},
            }
        else:
            payload["tool_choice"] = "auto"

    response = requests.post(ENDPOINT, json=payload, stream=True, timeout=30)

    if response.status_code >= 400:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {"error": {"message": response.text}}

        error_message = error_payload.get("error", {}).get("message", response.text)
        raise requests.HTTPError(
            f"{response.status_code} Server Error: {error_message}", response=response
        )

    content_chunks = []
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason = "stop"

    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break

                chunk = json.loads(data_str)
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                # Collect content
                if delta.get("content"):
                    content_chunks.append(delta["content"])
                    print(delta["content"], end="", flush=True)

                # Collect tool calls
                if delta.get("tool_calls"):
                    tool_calls = delta["tool_calls"]

                # Get finish reason
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    content = "".join(content_chunks)
    return content, tool_calls, finish_reason


def _run_local_tool(tool_name: str, args: dict) -> str:
    """Simulate local execution of the example tools."""
    if tool_name == "get_weather":
        units = args.get("units") or "celsius"
        unit_suffix = "C" if units == "celsius" else "F"
        return f"The weather in {args['location']} is sunny and 22°{unit_suffix}"
    if tool_name == "get_stock_price":
        mock_prices = {
            "AAPL": "$175.43",
            "GOOGL": "$142.87",
            "MSFT": "$378.91",
            "TSLA": "$242.15",
        }
        symbol = args["symbol"].upper()
        return f"The current price of {symbol} is {mock_prices.get(symbol, '$100.00')}"
    return "Tool result"


def main():
    """Run example streaming tool calling interactions."""
    print("=" * 60)
    print("Streaming Tool Calling Example with m serve")
    print("=" * 60)

    # Example 1: Request that should trigger weather tool
    print("\n1. Weather Query (Streaming)")
    print("-" * 60)
    messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]

    print(f"User: {messages[0]['content']}")
    print("\nAssistant: ", end="", flush=True)
    content, tool_calls, finish_reason = make_streaming_request(
        messages, tools=tools, tool_name="get_weather"
    )

    print(f"\n\nFinish Reason: {finish_reason}")

    if tool_calls:
        print("\nTool Calls:")
        for tool_call in tool_calls:
            func = tool_call["function"]
            args = json.loads(func["arguments"])
            print(f"  - {func['name']}({json.dumps(args)})")
    elif content:
        print("(Content already displayed above)")
    else:
        print("Assistant returned no content and no tool calls.")

    # Example 2: Request that should trigger stock price tool
    print("\n\n2. Stock Price Query (Streaming)")
    print("-" * 60)
    messages = [{"role": "user", "content": "What's the current stock price of AAPL?"}]

    print(f"User: {messages[0]['content']}")
    print("\nAssistant: ", end="", flush=True)
    content, tool_calls, finish_reason = make_streaming_request(
        messages, tools=tools, tool_name="get_stock_price"
    )

    print(f"\n\nFinish Reason: {finish_reason}")

    if tool_calls:
        print("\nTool Calls:")
        for tool_call in tool_calls:
            func = tool_call["function"]
            args = json.loads(func["arguments"])
            print(f"  - {func['name']}({json.dumps(args)})")
    elif content:
        print("(Content already displayed above)")
    else:
        print("Assistant returned no content and no tool calls.")

    # Example 3: Request without tools (normal chat)
    print("\n\n3. Normal Chat (No Tools, Streaming)")
    print("-" * 60)
    messages = [{"role": "user", "content": "Hello! How are you?"}]

    print(f"User: {messages[0]['content']}")
    print("\nAssistant: ", end="", flush=True)
    content, tool_calls, finish_reason = make_streaming_request(messages, tools=None)

    print(f"\n\nFinish Reason: {finish_reason}")

    # Example 4: Multi-turn conversation with tool use
    print("\n\n4. Multi-turn Conversation (Streaming)")
    print("-" * 60)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris?"}
    ]

    print(f"User: {messages[0]['content']}")
    print("\nAssistant: ", end="", flush=True)
    content, tool_calls, finish_reason = make_streaming_request(
        messages, tools=tools, tool_name="get_weather"
    )
    print()  # New line after streaming

    if tool_calls:
        print("\nAssistant requested tool calls:")

        # Add assistant message once before processing tool calls
        messages.append(
            {
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": tool_calls,
            }
        )

        tool_results: list[str] = []

        # Process each tool call and add tool responses
        for tool_call in tool_calls:
            func = tool_call["function"]
            args = json.loads(func["arguments"])
            print(f"  - {func['name']}({json.dumps(args)})")

            tool_result = _run_local_tool(func["name"], args)
            tool_results.append(tool_result)
            print(f"    Result: {tool_result}")

            # Add tool response to conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                }
            )

        # Get final response after tool execution
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Original question: {messages[0]['content']}\n"
                    f"Tool result: {'; '.join(tool_results)}\n"
                    "Answer the original question directly using only that tool "
                    "result. Do not mention unrelated topics or other tools."
                ),
            }
        )
        print("\nGetting final response after tool execution...")
        print("Assistant: ", end="", flush=True)
        content, tool_calls, finish_reason = make_streaming_request(
            messages, tools=None
        )
        print()  # New line after streaming
        if not content:
            print("Assistant returned no content after tool execution.")
    elif content:
        print("(Content already displayed above)")
    else:
        print("Assistant returned no content and no tool calls.")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server.")
        print("Make sure the server is running:")
        print("  uv run m serve docs/examples/m_serve/m_serve_example_tool_calling.py")
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}")
        if e.response is not None:
            try:
                print("Server response:", json.dumps(e.response.json(), indent=2))
            except ValueError:
                print("Server response:", e.response.text)
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
