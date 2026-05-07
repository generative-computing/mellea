"""Client example for testing tool calling with m serve.

This script demonstrates how to interact with an m serve server
that supports tool calling using the OpenAI-compatible API.

Usage:
    1. Start the server:
       uv run m serve docs/examples/m_serve/m_serve_example_tool_calling.py

    2. Run this client:
       uv run python docs/examples/m_serve/client_tool_calling.py
"""

import json

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
                "RootModel": {
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
                }
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol",
            "parameters": {
                "RootModel": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock ticker symbol, e.g. AAPL, GOOGL",
                        }
                    },
                    "required": ["symbol"],
                }
            },
        },
    },
]


def make_request(
    messages: list[dict], tools: list[dict] | None = None, tool_name: str | None = None
) -> dict:
    """Make a request to the m serve API.

    Args:
        messages: List of message dictionaries
        tools: Optional list of tool definitions
        tool_name: Optional tool name to request explicitly

    Returns:
        Response dictionary from the API
    """
    payload = {
        "model": "gpt-3.5-turbo",  # Model name (not used by m serve)
        "messages": messages,
        "temperature": 0.7,
    }

    if tools:
        payload["tools"] = tools
        if tool_name is not None:
            # m serve forwards tool_choice to compatible backends, but the
            # downstream provider/model may ignore it or treat it as a weak
            # preference rather than a guarantee. Use an explicit function
            # selection in this client so the example demonstrates the API
            # contract even when the model would otherwise decline to call tools.
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_name},
            }
        else:
            payload["tool_choice"] = "auto"

    response = requests.post(ENDPOINT, json=payload, timeout=30)

    if response.status_code >= 400:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {"error": {"message": response.text}}

        error_message = error_payload.get("error", {}).get("message", response.text)
        raise requests.HTTPError(
            f"{response.status_code} Server Error: {error_message}", response=response
        )

    return response.json()


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
    """Run example tool calling interactions."""
    print("=" * 60)
    print("Tool Calling Example with m serve")
    print("=" * 60)

    # Example 1: Request that should trigger weather tool
    print("\n1. Weather Query")
    print("-" * 60)
    messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]

    print(f"User: {messages[0]['content']}")
    response = make_request(messages, tools=tools, tool_name="get_weather")

    choice = response["choices"][0]
    print(f"\nFinish Reason: {choice['finish_reason']}")

    if choice.get("message", {}).get("tool_calls"):
        print("\nTool Calls:")
        for tool_call in choice["message"]["tool_calls"]:
            func = tool_call["function"]
            args = json.loads(func["arguments"])
            print(f"  - {func['name']}({json.dumps(args)})")
    elif choice.get("message", {}).get("content"):
        print(f"Assistant: {choice['message']['content']}")
    else:
        print("Assistant returned no content and no tool calls.")

    # Example 2: Request that should trigger stock price tool
    print("\n\n2. Stock Price Query")
    print("-" * 60)
    messages = [{"role": "user", "content": "What's the current stock price of AAPL?"}]

    print(f"User: {messages[0]['content']}")
    response = make_request(messages, tools=tools, tool_name="get_stock_price")

    choice = response["choices"][0]
    print(f"\nFinish Reason: {choice['finish_reason']}")

    if choice.get("message", {}).get("tool_calls"):
        print("\nTool Calls:")
        for tool_call in choice["message"]["tool_calls"]:
            func = tool_call["function"]
            args = json.loads(func["arguments"])
            print(f"  - {func['name']}({json.dumps(args)})")
    elif choice.get("message", {}).get("content"):
        print(f"Assistant: {choice['message']['content']}")
    else:
        print("Assistant returned no content and no tool calls.")

    # Example 3: Request without tools (normal chat)
    print("\n\n3. Normal Chat (No Tools)")
    print("-" * 60)
    messages = [{"role": "user", "content": "Hello! How are you?"}]

    print(f"User: {messages[0]['content']}")
    response = make_request(messages, tools=None)

    choice = response["choices"][0]
    print(f"\nFinish Reason: {choice['finish_reason']}")
    print(f"Assistant: {choice['message']['content']}")

    # Example 4: Multi-turn conversation with tool use
    print("\n\n4. Multi-turn Conversation")
    print("-" * 60)
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]

    print(f"User: {messages[0]['content']}")
    response = make_request(messages, tools=tools, tool_name="get_weather")

    choice = response["choices"][0]
    assistant_message = choice["message"]

    if assistant_message.get("tool_calls"):
        print("\nAssistant requested tool calls:")

        # Add assistant message once before processing tool calls
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.get("content"),
                "tool_calls": assistant_message["tool_calls"],
            }
        )

        tool_results: list[str] = []

        # Process each tool call and add tool responses
        for tool_call in assistant_message["tool_calls"]:
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

        # Get final response after tool execution.
        # Ask for a concise answer that explicitly uses the tool result so the
        # example output includes the actual weather/price instead of only a
        # conversational acknowledgement.
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
        response = make_request(messages, tools=None)
        choice = response["choices"][0]
        if choice.get("message", {}).get("content"):
            print(f"Assistant: {choice['message']['content']}")
        else:
            print("Assistant returned no content after tool execution.")
    elif assistant_message.get("content"):
        print(f"Assistant: {assistant_message['content']}")
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
        print(f"Error: {e}")
