"""Responses API client for the tool-calling m serve example.

Demonstrates function-call output items returned by /v1/responses.
The response output[] array contains ResponseFunctionCall items when
the model requests a tool, which the client handles locally and sends
back as a follow-up request.

Usage:
    1. Start the server:
       uv run m serve docs/examples/m_serve/tool-calling/m_serve_example_tool_calling.py

    2. Run this client:
       uv run python docs/examples/m_serve/tool-calling/client_responses_tool_calling.py
"""

import json

import requests

BASE_URL = "http://localhost:8080"
ENDPOINT = f"{BASE_URL}/v1/responses"

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
    response = requests.post(ENDPOINT, json=payload, timeout=30)
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
    """Run example Responses API tool-calling interactions."""
    print("=" * 60)
    print("Tool Calling Example with /v1/responses")
    print("=" * 60)

    # --- Example 1: weather tool ---
    print("\n1. Weather Query")
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

    function_calls = [
        item for item in resp["output"] if item["type"] == "function_call"
    ]
    text_items = [item for item in resp["output"] if item["type"] == "message"]

    if function_calls:
        print("\nFunction calls:")
        for fc in function_calls:
            args = json.loads(fc["arguments"])
            print(f"  - {fc['name']}({json.dumps(args)})")
    elif text_items:
        print(f"Assistant: {resp['output_text']}")
    else:
        print("Assistant returned no output.")

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

    function_calls = [
        item for item in resp["output"] if item["type"] == "function_call"
    ]
    if function_calls:
        print("\nFunction calls:")
        for fc in function_calls:
            args = json.loads(fc["arguments"])
            print(f"  - {fc['name']}({json.dumps(args)})")
    else:
        print(f"Assistant: {resp['output_text']}")

    # --- Example 3: tool call + follow-up ---
    print("\n\n3. Tool Call + Follow-up")
    print("-" * 60)
    question = "What's the weather in Paris?"
    print(f"User: {question}")

    resp = make_request(
        {
            "model": "granite4.1:3b",
            "input": question,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
    )

    function_calls = [
        item for item in resp["output"] if item["type"] == "function_call"
    ]
    if function_calls:
        print("\nAssistant requested tool calls:")
        tool_results = []
        for fc in function_calls:
            args = json.loads(fc["arguments"])
            print(f"  - {fc['name']}({json.dumps(args)})")
            result = _run_local_tool(fc["name"], args)
            tool_results.append(result)
            print(f"    Result: {result}")

        follow_up = (
            f"Original question: {question}\n"
            f"Tool result: {'; '.join(tool_results)}\n"
            "Answer the original question directly using only that tool result."
        )
        print("\nGetting final response after tool execution...")
        resp2 = make_request({"model": "granite4.1:3b", "input": follow_up})
        print(f"Assistant: {resp2['output_text']}")
    else:
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
            " docs/examples/m_serve/tool-calling/m_serve_example_tool_calling.py"
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
