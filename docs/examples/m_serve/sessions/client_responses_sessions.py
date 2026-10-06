# pytest: skip_always
"""Multi-turn session client using the Responses API.

Demonstrates server-side session storage via `previous_response_id`.
Each request only sends the new user turn; the server prepends the prior
conversation history automatically.

Contrast with `/v1/chat/completions` where the client must resend the
full message history on every request.

Usage:
    1. Start the server:
       m serve docs/examples/m_serve/sessions/m_serve_example_sessions.py

    2. Run this client:
       python docs/examples/m_serve/sessions/client_responses_sessions.py
"""

import openai

PORT = 8080

client = openai.OpenAI(api_key="na", base_url=f"http://0.0.0.0:{PORT}/v1")

MODEL = "granite4.1:3b"

# ── Turn 1: ask a question ──────────────────────────────────────────────────

response1 = client.responses.create(
    model=MODEL, input="My name is Alice and I love astronomy."
)

print("Turn 1:")
print(f"  Response ID : {response1.id}")
print(f"  Reply       : {response1.output_text}")
print()

# ── Turn 2: follow-up — only send the new question, no history ──────────────
# The server looks up the history stored under response1.id and prepends it
# before calling the model.  The model sees both turns even though we only
# sent one message here.

response2 = client.responses.create(
    model=MODEL,
    input="What is my name and what do I love?",
    previous_response_id=response1.id,
)

print("Turn 2:")
print(f"  Response ID : {response2.id}")
print(f"  Reply       : {response2.output_text}")
print()

# ── Turn 3: continue the chain ──────────────────────────────────────────────

response3 = client.responses.create(
    model=MODEL,
    input="Recommend one beginner telescope for me.",
    previous_response_id=response2.id,
)

print("Turn 3:")
print(f"  Response ID : {response3.id}")
print(f"  Reply       : {response3.output_text}")
print()

# ── Retrieve a past response by ID ──────────────────────────────────────────

import httpx

retrieved = httpx.get(f"http://0.0.0.0:{PORT}/v1/responses/{response1.id}")
print(f"Retrieved response1 via GET /v1/responses/{response1.id}:")
print(f"  Status : {retrieved.json()['status']}")
print(f"  Text   : {retrieved.json()['output_text']}")
