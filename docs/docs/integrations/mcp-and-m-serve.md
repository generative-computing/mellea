---
title: "MCP and m serve"
description: "Expose Mellea programs as MCP tools with FastMCP, or serve them as an OpenAI-compatible endpoint with m serve."
# diataxis: how-to
---

# MCP and m serve

Mellea programs are Python programs. You can expose them to the outside world in two ways:

- **MCP** — wrap Mellea functions as [Model Context Protocol](https://modelcontextprotocol.io/) tools, callable by any MCP client (Claude Desktop, Cursor, etc.)
- **`m serve`** — run a Mellea program as an OpenAI-compatible chat endpoint, so other LLM clients can call it as if it were a model

## MCP integration

**Prerequisites:** `pip install mellea`, `pip install "mcp[cli]"`, Ollama running locally.

Mellea integrates with MCP via [FastMCP](https://github.com/jlowin/fastmcp): you wrap Mellea functions as MCP tools, then expose them to any MCP-compatible client.

### Creating an MCP server

```python
from mcp.server.fastmcp import FastMCP
from mellea import MelleaSession
from mellea.backends import ModelOption, model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import Requirement
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

mcp = FastMCP("mellea-demo")

@mcp.tool()
def write_a_poem(word_limit: int) -> str:
    """Write a poem with a specified word limit."""
    m = MelleaSession(
        OllamaModelBackend(
            model_ids.IBM_GRANITE_4_HYBRID_MICRO,
            model_options={ModelOption.MAX_NEW_TOKENS: word_limit + 10},
        )
    )
    word_limit_req = Requirement(
        f"Use only {word_limit} words.",
        validation_fn=simple_validate(lambda x: len(x.split()) < word_limit),
    )
    result = m.instruct(
        "Write a poem.",
        requirements=[word_limit_req],
        strategy=RejectionSamplingStrategy(loop_budget=2),
    )
    return str(result.value)

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting."""
    return f"Hello, {name}!"
```

Each `@mcp.tool()` function becomes a tool that MCP clients can call. The docstring is used as the tool description — write it clearly. Mellea's requirements and sampling strategies work exactly as they do in regular code; the MCP layer just wraps the result.

### Running the server

Start the MCP dev UI to test your server interactively:

```bash
uv run mcp dev your_server.py
```

This opens a browser-based inspector at `http://localhost:5173` where you can call tools, inspect arguments, and see outputs.

To run the server directly:

```bash
uv run your_server.py
```

### Multiple tools in one server

A single `FastMCP` server can expose multiple tools, resources, and prompts:

```python
from mcp.server.fastmcp import FastMCP
from mellea import MelleaSession, generative, start_session
from mellea.backends.ollama import OllamaModelBackend
from typing import Literal

mcp = FastMCP("mellea-tools")

@mcp.tool()
def summarize(text: str, max_words: int = 100) -> str:
    """Summarize the provided text."""
    m = MelleaSession(OllamaModelBackend())
    result = m.instruct(
        "Summarize the following text in {{max_words}} words or fewer: {{text}}",
        user_variables={"text": text, "max_words": str(max_words)},
    )
    return str(result)

@mcp.tool()
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of the text as positive, negative, or neutral."""
    @generative
    def _classify(text: str) -> Literal["positive", "negative", "neutral"]:
        """Classify sentiment."""
        ...

    m = start_session()
    return _classify(m, text=text)
```

> **Note:** Each tool invocation creates a new `MelleaSession`. For high-throughput servers, consider reusing sessions across calls by initializing them at module level. **Full example:** [`docs/examples/notebooks/mcp_example.ipynb`](../../examples/notebooks/mcp_example.ipynb)

## m serve — OpenAI-compatible endpoint

**Prerequisites:** `pip install mellea`.

`m serve` runs any Mellea program as an OpenAI-compatible chat endpoint. This lets other LLM clients (LangChain, OpenAI SDK, curl) call your program as if it were a model.

### The serve() function

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

`m serve` loads your file, finds `serve()`, and routes incoming requests to it. `ChatMessage` has `role` and `content` fields matching the OpenAI chat format.

### Example serve program

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

### Starting m serve

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

### Calling the served endpoint

Any OpenAI-compatible client works. Using `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarize this in one sentence."}]}'
```

> **Full example:** [`docs/examples/m_serve/m_serve_example_simple.py`](../../examples/m_serve/m_serve_example_simple.py)

---

**Previous:** [AWS Bedrock and IBM watsonx](./bedrock-and-watsonx.md) |
**Next:** [Metrics and Telemetry](../evaluation-and-observability/metrics-and-telemetry.md)
