# pytest: skip_always
"""Responses API streaming client for the streaming m serve example.

Demonstrates the semantic SSE events emitted by /v1/responses when
stream=True. Events arrive as typed objects via the OpenAI SDK's
response stream iterator.

Usage:
    1. Start the server:
       m serve docs/examples/m_serve/streaming/m_serve_example_streaming.py

    2. Run this client:
       python docs/examples/m_serve/streaming/client_responses_streaming.py

Set `streaming` below to toggle between streaming and non-streaming mode.
"""

import openai

PORT = 8080

client = openai.OpenAI(api_key="na", base_url=f"http://0.0.0.0:{PORT}/v1")

streaming = True  # streaming enabled toggle

print(f"stream={streaming} response:")
print("-" * 50)

if streaming:
    with client.responses.stream(
        model="granite4.1:3b", input="Count down from 10 using words not digits."
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
else:
    response = client.responses.create(
        model="granite4.1:3b",
        input="Count down from 10 using words not digits.",
        stream=False,
    )
    print(response.output_text)

print("\n" + "-" * 50)
print("Stream complete!")
