"""Responses API client for the server-side tool-calling example.

Demonstrates the key difference from the client-side tool-calling example:
when tool execution happens inside ``serve()``, the server runs the full
agentic loop (call model → execute tools → re-prompt) before returning.
The client sends a single request and receives a fully-resolved answer —
no ``function_call`` items to handle, no follow-up request needed.

Compare with ``client_responses_tool_calling.py``, which handles the
``function_call`` items itself and makes a second request to get the
final answer.

Usage:
    1. Start the server:
       uv run m serve docs/examples/m_serve/tool-calling/m_serve_example_tool_calling_server_side.py

    2. Run this client:
       uv run python docs/examples/m_serve/tool-calling/client_responses_tool_calling_server_side.py
"""

import json

import requests

BASE_URL = "http://localhost:8080"
ENDPOINT = f"{BASE_URL}/v1/responses"

# Tool definitions are still sent so the server knows which tools are available
# for this request.  The server executes them — the client never calls them.
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


def make_request(payload: dict) -> dict:
    """POST to /v1/responses and return the parsed response body."""
    response = requests.post(ENDPOINT, json=payload, timeout=60)
    if response.status_code >= 400:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {"error": {"message": response.text}}
        message = error_payload.get("error", {}).get("message", response.text)
        raise requests.HTTPError(
            f"{response.status_code} Server Error: {message}", response=response
        )
    return response.json()


def main():
    """Run server-side tool-calling examples with /v1/responses."""
    print("=" * 60)
    print("Server-Side Tool Calling with /v1/responses")
    print("=" * 60)
    print(
        "\nIn this example the server runs the tool execution loop.\n"
        "Each request returns a fully-resolved answer — no client\n"
        "loop or follow-up request required.\n"
    )

    # --- Example 1: weather tool (server executes, client gets final answer) ---
    print("1. Weather Query")
    print("-" * 60)
    question = "What's the weather like in Tokyo?"
    print(f"User: {question}")

    resp = make_request(
        {
            "model": "granite4.1:3b",
            "input": question,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
    )

    # The server has already executed get_weather and re-prompted the model.
    # output[] contains only the final message — no function_call items.
    function_calls = [
        item for item in resp["output"] if item["type"] == "function_call"
    ]
    if function_calls:
        # This should not happen with the server-side module; shown for clarity.
        print("  (unexpected) Server returned unexecuted function_call items:")
        for fc in function_calls:
            print(f"    - {fc['name']}({fc['arguments']})")
    else:
        print(f"Assistant: {resp['output_text']}")

    # --- Example 2: stock price tool ---
    print("\n\n2. Stock Price Query")
    print("-" * 60)
    question = "What's the current stock price of AAPL?"
    print(f"User: {question}")

    resp = make_request(
        {
            "model": "granite4.1:3b",
            "input": question,
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_stock_price"},
            },
        }
    )

    print(f"Assistant: {resp['output_text']}")

    # --- Example 3: no tools — plain chat, unchanged behaviour ---
    print("\n\n3. Plain Chat (no tools)")
    print("-" * 60)
    question = "What is the capital of France?"
    print(f"User: {question}")

    resp = make_request({"model": "granite4.1:3b", "input": question})

    print(f"Assistant: {resp['output_text']}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server.")
        print("Make sure the server is running:")
        print(
            "  uv run m serve"
            " docs/examples/m_serve/tool-calling/m_serve_example_tool_calling_server_side.py"
        )
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}")
        if e.response is not None:
            try:
                print("Server response:", json.dumps(e.response.json(), indent=2))
            except ValueError:
                print("Server response:", e.response.text)
    except Exception as e:
        print(f"Error: {e}")
