---
title: "m serve"
description: "Run a Mellea program as an OpenAI-compatible chat endpoint with m serve."
# diataxis: how-to
---

`m serve` runs any Mellea program as an OpenAI-compatible server. This lets
any LLM client — LangChain, the OpenAI SDK, `curl` — call your Mellea program as if
it were a model. Both the Chat Completions API and the Responses API are supported.

**Prerequisites:** `pip install "mellea[server]"`.

## The serve() function

Your program must define a `serve()` function with this signature:

```python
from mellea.core import ModelOutputThunk, SamplingResult
from mellea.serve import ChatMessage

def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict | None = None,
) -> ModelOutputThunk | SamplingResult:
    """Your Mellea program logic here."""
    ...
```

`m serve` loads your file, finds `serve()`, and routes incoming requests to it.
`ChatMessage` has `role` and `content` fields matching the OpenAI chat format.

## Example serve program

```python
import mellea
from mellea.core import ModelOutputThunk, Requirement, SamplingResult
from mellea.serve import ChatMessage
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

session = mellea.start_session(ctx=ChatContext())

def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict | None = None,
) -> ModelOutputThunk | SamplingResult:
    """Takes a prompt as input and runs it through a Mellea program."""
    message = input[-1].content
    reqs = [
        Requirement(
            "Keep this under 50 words",
            validation_fn=simple_validate(lambda x: len(x.split()) < 50),
        ),
        *(requirements or []),
    ]
    return session.instruct(
        description=message,
        requirements=reqs,
        strategy=RejectionSamplingStrategy(loop_budget=3),
        model_options=model_options,
    )
```

The session is initialised at module level so it is reused across requests. This
preserves the `ChatContext` conversation history across turns.

## Starting m serve

```bash
m serve path/to/your_program.py
```

The server starts on port 8000 by default and exposes:

- `POST /v1/chat/completions` — OpenAI Chat Completions API
- `POST /v1/responses` — OpenAI Responses API
- `GET /health` — health check

To see all options:

```bash
m serve --help
```

## Calling the served endpoint

Both endpoints are served by the same `serve()` function. The Responses API
input is converted to `ChatMessage` format before being passed to `serve()`,
so no changes to your program are needed.

### Chat Completions

Any OpenAI-compatible client works. Using `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mellea", "messages": [{"role": "user", "content": "Summarize this in one sentence."}]}'
```

Using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="mellea",
    messages=[{"role": "user", "content": "Summarize this in one sentence."}],
)
print(response.choices[0].message.content)
```

**Full example:** [`docs/examples/m_serve/simple/m_serve_example_simple.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/m_serve/simple/m_serve_example_simple.py)

### Responses API

Using `curl`:

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "mellea", "input": "Summarize this in one sentence."}'
```

Using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.responses.create(
    model="mellea",
    input="Summarize this in one sentence.",
)
print(response.output_text)
```

Streaming with semantic events:

```python
stream = client.responses.create(
    model="mellea",
    input="Tell me a story.",
    stream=True,
)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

---

**See also:** [Context and Sessions](../concepts/context-and-sessions.md) |
[Backends and Configuration](../how-to/backends-and-configuration.md) |
[CLI Reference](../reference/cli.md)
