# MCP Examples

Examples demonstrating the Model Context Protocol (MCP) in both directions:
exposing a Mellea program as an MCP server, and consuming tools from an MCP
server inside Mellea. For the API overview, see
[MCP integration](../../docs/integrations/mcp.md) and the
[MCP tools section](../../docs/how-to/tools-and-agents.md#mcp-tools) of the
Tools and Agents how-to.

## Examples

| File | Description |
| :--- | :--- |
| [`mcp_example.py`](./mcp_example.py) | **Expose Mellea as an MCP server.** Makes a Mellea instruct-validate-repair loop callable as an MCP tool from Claude Desktop, Langflow, or any MCP client. |
| [`github_activity_summary.py`](./github_activity_summary.py) | **Consume MCP server tools from Mellea.** Discovers tools on a remote MCP server and drops them into a Mellea `react()` loop. Needs `GITHUB_TOKEN`. |
| [`mcp_server.json`](./mcp_server.json) | MCP client config that launches `mcp_example.py` over stdio. Replace `<ABSOLUTE PATH>` so that `--directory` points at this directory. |
