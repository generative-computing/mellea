# pytest: skip_always
"""Example client demonstrating responses from m serve.

This example shows how to use the OpenAI Python client with a Mellea server
started with:

    m serve docs/examples/m_serve/m_serve_example_streaming.py

Set ``streaming`` below to:
- ``True`` for incremental SSE chunks
- ``False`` for a normal non-streaming response
"""

import openai

PORT = 8080

client = openai.OpenAI(api_key="na", base_url=f"http://0.0.0.0:{PORT}/v1")

streaming = True  # streaming enabled toggle

print(f"stream={streaming} response:")
print("-" * 50)

# Request either a streaming or non-streaming response from the dedicated example server
if streaming:
    stream_result = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "Count down from 100 using words not digits."}
        ],
        model="granite4:micro-h",
        stream=True,
    )
    for chunk in stream_result:
        if chunk.choices[0].delta.content:
            # If you want to see the chunks more clearly separated, change end
            print(chunk.choices[0].delta.content, end="", flush=True)
else:
    completion_result = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "Count down from 100 using words not digits."}
        ],
        model="granite4:micro-h",
        stream=False,
    )
    print(completion_result.choices[0].message.content)

print("\n" + "-" * 50)
print("Stream complete!")
