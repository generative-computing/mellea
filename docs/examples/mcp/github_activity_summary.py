# pytest: skip_always
"""Example: summarise recent GitHub activity using the GitHub MCP server.

Demonstrates the mellea MCP workflow:
  1. Discover all tools on the server with discover_mcp_tools()
  2. Pick only the ones needed by name
  3. Drive multi-turn tool use with mellea's react() loop

Prerequisites:
    pip install 'mellea[tools]'
    export GITHUB_TOKEN=<token with repo + read:user scopes>

Usage:
    uv run python docs/examples/mcp/github_activity_summary.py
"""

import argparse
import asyncio
import os
from datetime import UTC, datetime, timedelta

from mellea import start_session
from mellea.backends import model_ids
from mellea.core.base import AbstractMelleaTool
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from mellea.stdlib.tools.mcp import discover_mcp_tools, http_connection

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
TOOLS_NEEDED = {"get_me", "search_pull_requests"}


async def main(days: int) -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN environment variable is required")

    now = datetime.now(UTC)
    since = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    today = now.strftime("%Y-%m-%d")

    connection = http_connection(GITHUB_MCP_URL, api_key=token)
    m = start_session(model_id=model_ids.IBM_GRANITE_4_1_8B)

    # --- Tool discovery ---
    specs = await discover_mcp_tools(connection)
    print(f"Discovered {len(specs)} tools on the GitHub MCP server")

    # --- Tool selection ---
    relevant = [s for s in specs if s.name in TOOLS_NEEDED]
    print(f"Using {len(relevant)} tools: {[s.name for s in relevant]}")
    tools: list[AbstractMelleaTool] = [s.as_mellea_tool() for s in relevant]

    # --- Agent loop ---
    result, _ = await react(
        goal=(
            f"Today is {today}. Find my GitHub username, then search for pull requests "
            f"I authored since {since} filtering by my username. "
            "List each pull request with its title, number, and repository."
        ),
        context=ChatContext(),
        backend=m.backend,
        tools=tools,
        loop_budget=6,
    )

    print("\n--- Activity Summary ---")
    print(result.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days", type=int, default=14, help="How many days back to look"
    )
    args = parser.parse_args()
    asyncio.run(main(args.days))
