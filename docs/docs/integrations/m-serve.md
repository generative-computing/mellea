---
title: "m serve"
description: "Run a Mellea program as an OpenAI-compatible chat endpoint with m serve."
# diataxis: how-to
---

# m serve

`m serve` runs any Mellea program as an OpenAI-compatible chat endpoint. This lets
any LLM client — LangChain, the OpenAI SDK, `curl` — call your Mellea program as if
it were a model.

**Prerequisites:** `pip install mellea`.

## The serve() function

Your program must define a `serve()` function with this signature:

```python
from cli.serve.models import ChatMessage
from mellea.core import ModelOutputThunk, SamplingResult

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
from cli.serve.models import ChatMessage
from mellea.core import ModelOutputThunk, Requirement, SamplingResult
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

- `POST /v1/chat/completions` — OpenAI-compatible chat completions endpoint
- `GET /health` — health check

To see all options:

```bash
m serve --help
```

## Calling the served endpoint

Any OpenAI-compatible client works. Using `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarize this in one sentence."}]}'
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

**Full example:** [`docs/examples/m_serve/m_serve_example_simple.py`](../../examples/m_serve/m_serve_example_simple.py)

---

**Previous:** [MCP Integration](./mcp.md) |
**Next:** [LangChain and smolagents](./langchain-and-smolagents.md)

**See also:** [Context and Sessions](../concepts/context-and-sessions.md) |
[Backends and Configuration](../guide/backends-and-configuration.md)
