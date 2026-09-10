# pytest: skip_always
"""Responses API client for the simple m serve example.

Usage:
    1. Start the server:
       m serve docs/examples/m_serve/simple/m_serve_example_simple.py

    2. Run this client:
       python docs/examples/m_serve/simple/client_responses.py
"""

import openai

PORT = 8080

client = openai.OpenAI(api_key="na", base_url=f"http://0.0.0.0:{PORT}/v1")

response = client.responses.create(
    model="granite4.1:3b", input="Find all the real roots of x^3 + 1."
)

print(response.output_text)
