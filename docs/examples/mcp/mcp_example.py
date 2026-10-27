# pytest: skip_always
"""Exposing a Mellea instruct-validate-repair loop as an MCP server.

Demonstrates a FastMCP server whose `write_a_poem` tool runs `m.instruct()` with a
word-limit `Requirement` and `RejectionSamplingStrategy`, so the tool validates and
repairs its own output before returning. The same pattern makes any Mellea program
callable from Claude Desktop, Langflow, Cursor, or any other MCP client. The file
also registers a `greeting://{name}` resource.

To use it from Langflow, follow https://docs.langflow.org/mcp-client#mcp-stdio-mode
and register the server with the config in `mcp_server.json` in this directory,
replacing `<ABSOLUTE PATH>` with the absolute path to your Mellea checkout. The
config launches `uv --directory <path>/docs/examples/mcp run mcp run mcp_example.py`
over stdio.

For a walkthrough of building servers, see docs/docs/integrations/mcp.md.

Prerequisites:
    uv pip install "mcp[cli]"
    ollama pull smollm2:1.7b
    ollama serve

Usage:
    uv run mcp dev docs/examples/mcp/mcp_example.py
"""

import io
from contextlib import redirect_stdout

from mcp.server.fastmcp import FastMCP

from mellea import MelleaSession
from mellea.backends import ModelOption, model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import MelleaLogger, ModelOutputThunk, Requirement
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Create an MCP server
mcp = FastMCP("Demo")


@mcp.tool()
def write_a_poem(word_limit: int) -> str:
    """Write a poem with a word limit."""
    collect_stdout = io.StringIO()
    with redirect_stdout(collect_stdout):
        # TODO: the mcp tool still collects all logging.info and tries to forward it -- not sure how to turn this off

        m = MelleaSession(
            OllamaModelBackend(
                model_ids.HF_SMOLLM2_2B,
                model_options={ModelOption.MAX_NEW_TOKENS: word_limit + 10},
            )
        )
        wl_req = Requirement(
            f"Use only {word_limit} words.",
            validation_fn=simple_validate(lambda x: len(x.split(" ")) < word_limit),
        )

        res = m.instruct(
            "Write a poem",
            requirements=[wl_req],
            strategy=RejectionSamplingStrategy(loop_budget=2),
        )

    collect_stdout.flush()
    assert isinstance(res, ModelOutputThunk)
    return str(res.value)


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting."""
    return f"Hello, {name}!"
