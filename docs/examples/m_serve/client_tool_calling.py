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
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def make_request(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Make a request to the m serve API.

    Args:
        messages: List of message dictionaries
        tools: Optional list of tool definitions

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
        payload["tool_choice"] = "auto"

    response = requests.post(ENDPOINT, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


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
    response = make_request(messages, tools=tools)

    choice = response["choices"][0]
    print(f"\nFinish Reason: {choice['finish_reason']}")

    if choice.get("message", {}).get("tool_calls"):
        print("\nTool Calls:")
        for tool_call in choice["message"]["tool_calls"]:
            func = tool_call["function"]
            args = json.loads(func["arguments"])
            print(f"  - {func['name']}({json.dumps(args)})")
    else:
        print(f"Assistant: {choice['message']['content']}")

    # Example 2: Request that should trigger calculator tool
    print("\n\n2. Math Query")
    print("-" * 60)
    messages = [{"role": "user", "content": "What is 15 * 23 + 7?"}]

    print(f"User: {messages[0]['content']}")
    response = make_request(messages, tools=tools)

    choice = response["choices"][0]
    print(f"\nFinish Reason: {choice['finish_reason']}")

    if choice.get("message", {}).get("tool_calls"):
        print("\nTool Calls:")
        for tool_call in choice["message"]["tool_calls"]:
            func = tool_call["function"]
            args = json.loads(func["arguments"])
            print(f"  - {func['name']}({json.dumps(args)})")
    else:
        print(f"Assistant: {choice['message']['content']}")

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
    response = make_request(messages, tools=tools)

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

        # Process each tool call and add tool responses
        for tool_call in assistant_message["tool_calls"]:
            func = tool_call["function"]
            args = json.loads(func["arguments"])
            print(f"  - {func['name']}({json.dumps(args)})")

            # Simulate tool execution
            if func["name"] == "get_weather":
                tool_result = f"The weather in {args['location']} is sunny and 22°C"
            else:
                tool_result = "Tool result"

            # Add tool response to conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                }
            )

        # Get final response after tool execution
        print("\nGetting final response after tool execution...")
        response = make_request(messages, tools=tools)
        choice = response["choices"][0]
        print(f"Assistant: {choice['message']['content']}")

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
    except Exception as e:
        print(f"Error: {e}")
