---
title: "MCP Integration"
description: "Expose Mellea functions as Model Context Protocol tools, callable from Claude Desktop, Cursor, and any MCP-compatible client."
# diataxis: how-to
---

[Model Context Protocol](https://modelcontextprotocol.io/) (MCP) is an open standard
for exposing tools to AI clients. Mellea integrates with MCP via
[FastMCP](https://github.com/jlowin/fastmcp): wrap any Mellea function as an MCP tool
and call it from Claude Desktop, Cursor, or any MCP-compatible client.

**Prerequisites:** `pip install mellea`, `pip install "mcp[cli]"`, Ollama running locally.

## Creating an MCP server

Decorate any function with `@mcp.tool()`. The docstring becomes the tool description
visible to the AI client.

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

Each `@mcp.tool()` function becomes a callable tool. Mellea's requirements and
sampling strategies work exactly as they do in regular code — the MCP layer just
wraps the result.

## Multiple tools in one server

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

> **Note:** Each tool invocation creates a new `MelleaSession`. For high-throughput
> servers, consider initializing sessions at module level and reusing them across calls.

## Running the server

Start the MCP dev UI to test interactively:

```bash
uv run mcp dev your_server.py
```

This opens a browser-based inspector at `http://localhost:5173` where you can call
tools, inspect arguments, and see outputs.

To run the server directly:

```bash
uv run your_server.py
```

**Full example:** [`docs/examples/notebooks/mcp_example.ipynb`](../../examples/notebooks/mcp_example.ipynb)

---

**See also:** [Backends and Configuration](../guide/backends-and-configuration)
