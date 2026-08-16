# pytest: skip_always

"""Client demonstrating how to interact with the model-routing m serve example.

Pattern A: use the standard `model` field to select the backend via client_options.
Pattern B: unknown model falls back to the default backend.

The allowlist and default behavior is an implementation decision in the example
Mellea program being served. It could easily be changed to ignore the requested
model or to throw an error instead of having a default model.

Run the server first:
    uv run m serve docs/examples/m_serve/model-routing/m_serve_example_model_routing.py
"""

import openai

PORT = 8080
client = openai.OpenAI(api_key="na", base_url=f"http://0.0.0.0:{PORT}/v1")

print("=== Pattern A: standard model field routes via client_options ===")
# The standard `model` field is read by the server via
# client_options and used to select the backend.
response_a = client.chat.completions.create(
    model="granite4.1:8b", messages=[{"role": "user", "content": "What is 2 + 2?"}]
)
print(f"model echoed back : {response_a.model}")
print(f"response          : {response_a.choices[0].message.content}\n")

print("=== Pattern B: unknown model falls back to default ===")
response_b = client.chat.completions.create(
    model="some-unknown-model", messages=[{"role": "user", "content": "What is 2 + 2?"}]
)
print(f"model echoed back : {response_b.model}")
print(f"response          : {response_b.choices[0].message.content}")
