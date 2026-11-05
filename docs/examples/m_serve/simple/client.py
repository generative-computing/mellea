# pytest: skip_always
"""Chat Completions client for the simple m serve example.

Usage:
    1. Start the server:
       m serve docs/examples/m_serve/simple/m_serve_example_simple.py

    2. Run this client:
       uv run python docs/examples/m_serve/simple/client.py
"""

import openai

PORT = 8080

client = openai.OpenAI(api_key="na", base_url=f"http://localhost:{PORT}/v1")

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Find all the real roots of x^3 + 1."}],
    model="granite4.1:3b",
)

print(response.choices[0])
