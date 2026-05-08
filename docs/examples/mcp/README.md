# MCP examples

Two directions are covered here:

- **Expose Mellea as an MCP server** — [`mcp_example.py`](mcp_example.py).
  Makes a Mellea instruct-validate-repair loop callable as an MCP tool from
  Claude Desktop, Langflow, or any MCP client.
- **Consume MCP server tools from Mellea** —
  [`github_activity_summary.py`](github_activity_summary.py). Discovers tools on
  a remote MCP server and drops them into a Mellea `react()` loop.

## Write a poem MCP (Mellea as server)

A simple example to show how to write a MCP tool with Mellea and
instruct-validate-repair. Being able to speak the tool language lets you integrate
with Claude Desktop, Langflow, and other MCP clients.

See code in [`mcp_example.py`](mcp_example.py).

### Running the poem server

Install the MCP SDK:

```bash
uv pip install "mcp[cli]"
```

Run the example in the MCP debug UI:

```bash
uv run mcp dev docs/examples/mcp/mcp_example.py
```

### Use in Langflow

Follow [this guide](https://docs.langflow.org/mcp-client#mcp-stdio-mode) to register
the tool. Insert the absolute path to the directory containing `mcp_example.py`:

```json
{
  "mcpServers": {
    "mellea_mcp_server": {
      "command": "uv",
      "args": [
        "--directory",
        "<ABSOLUTE PATH>/mellea/docs/examples/mcp",
        "run",
        "mcp",
        "run",
        "mcp_example.py"
      ]
    }
  }
}
```

## GitHub activity summary (Mellea as client)

Uses the hosted GitHub MCP server to summarize recent pull requests.
Demonstrates the two-step workflow (discover tools, pick the ones you need,
wrap as `MelleaTool`) and drives multi-turn tool use via `react()`.

See code in [`github_activity_summary.py`](github_activity_summary.py), and
the [Tools and Agents how-to guide](../../docs/how-to/tools-and-agents) for
the API overview.

### Running the activity summary

Install Mellea with tools support:

```bash
pip install 'mellea[tools]'
```

Set a GitHub token with `repo` and `read:user` scopes:

```bash
export GITHUB_TOKEN=<your token>
```

Run:

```bash
uv run python docs/examples/mcp/github_activity_summary.py --days 14
```

The script discovers every tool on the GitHub MCP server, filters down to
`get_me` and `search_pull_requests`, then asks the model to summarize your
pull-request activity over the specified window.
