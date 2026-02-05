"""Example of an MCP server.

You need to install the mcp package:
uv pip install "mcp[cli]"

and run the example in MCP debug UI:
uv run mcp dev docs/examples/mcp/mcp_example.py
"""

import io
from contextlib import redirect_stdout

from mcp.server.fastmcp import FastMCP

from mellea import MelleaSession
from mellea.backends import ModelOption, model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import FancyLogger, ModelOutputThunk, Requirement
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# #################
# run MCP debug UI with: uv run mcp dev docs/examples/tutorial/mcp_example.py
# ##################


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
