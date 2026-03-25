# pytest: skip_always
"""Example of using the OpenAI Python SDK with the mellea server.

This example shows how to use the official OpenAI Python SDK to interact
with a mellea backend through the OpenAI-compatible server.
"""

import openai


def test_openai_sdk_compatibility():
    """Test that the OpenAI SDK works with mellea server."""
    # Configure the OpenAI client to point to our mellea server
    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # mellea server doesn't require auth by default
    )

    # Make a simple chat completion request
    response = client.chat.completions.create(
        model="granite4:micro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        temperature=0.7,
        max_tokens=50,
    )

    # Access the response
    print(f"Response: {response.choices[0].message.content}")
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0


def test_streaming_with_openai_sdk():
    """Test streaming responses with the OpenAI SDK."""
    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

    # Make a streaming request
    stream = client.chat.completions.create(
        model="granite4:micro",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True,
        max_tokens=50,
    )

    # Collect streamed chunks
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)

    print()  # New line after streaming
    assert len(full_response) > 0


if __name__ == "__main__":
    print("Testing OpenAI SDK compatibility...")
    print("\n1. Basic completion:")
    test_openai_sdk_compatibility()

    print("\n2. Streaming completion:")
    test_streaming_with_openai_sdk()

    print("\nAll tests passed!")
