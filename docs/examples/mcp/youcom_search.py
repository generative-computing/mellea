# pytest: skip_always
"""Example: answer a question with live web sources using the You.com MCP server.

Demonstrates the mellea MCP workflow with a keyless remote server:
  1. Discover all tools on the server with discover_mcp_tools()
  2. Pick only the ones needed by name
  3. Drive multi-turn tool use with mellea's react() loop

The You.com free profile (https://api.you.com/mcp?profile=free) exposes a
`you-search` tool without any API key, so this example runs with no credentials.
Set YDC_API_KEY (from https://you.com/platform/api-keys) to use the
authenticated endpoint instead, which also exposes content extraction and
research tools.

Prerequisites:
    pip install 'mellea[tools]'
    Ollama running locally

Usage:
    uv run python docs/examples/mcp/youcom_search.py --query "What is the Mellea Python library?"
"""

import argparse
import asyncio
import os

from mellea import start_session
from mellea.backends import model_ids
from mellea.core.base import AbstractMelleaTool
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from mellea.stdlib.tools.mcp import discover_mcp_tools, http_connection

YOUCOM_MCP_URL = "https://api.you.com/mcp"
YOUCOM_FREE_PROFILE_URL = "https://api.you.com/mcp?profile=free"
TOOLS_NEEDED = {"you-search"}


async def main(query: str) -> None:
    api_key = os.environ.get("YDC_API_KEY")
    url = YOUCOM_MCP_URL if api_key else YOUCOM_FREE_PROFILE_URL

    connection = http_connection(url, api_key=api_key)
    m = start_session(model_id=model_ids.IBM_GRANITE_4_1_8B)

    # --- Tool discovery ---
    specs = await discover_mcp_tools(connection)
    print(f"Discovered {len(specs)} tools on the You.com MCP server")

    # --- Tool selection ---
    relevant = [s for s in specs if s.name in TOOLS_NEEDED]
    print(f"Using {len(relevant)} tools: {[s.name for s in relevant]}")
    tools: list[AbstractMelleaTool] = [s.as_mellea_tool() for s in relevant]

    # --- Agent loop ---
    result, _ = await react(
        goal=(
            f"Answer this question using web search: {query} "
            "Cite the source URLs you used."
        ),
        context=ChatContext(),
        backend=m.backend,
        tools=tools,
        loop_budget=6,
    )

    print("\n--- Answer ---")
    print(result.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="What is the Mellea Python library?",
        help="The question the agent should research",
    )
    args = parser.parse_args()
    asyncio.run(main(args.query))
