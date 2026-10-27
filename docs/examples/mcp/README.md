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
| [`mcp_example.py`](./mcp_example.py) | Mellea as the **server**: a FastMCP tool that runs an instruct-validate-repair loop, callable from Claude Desktop, Langflow, or any MCP client. Includes Langflow setup and run commands in its docstring. |
| [`mcp_server.json`](./mcp_server.json) | MCP client config that launches `mcp_example.py` over stdio. Replace `<ABSOLUTE PATH>` with your checkout path. |
| [`github_activity_summary.py`](./github_activity_summary.py) | Mellea as the **client**: discovers tools on the hosted GitHub MCP server, selects two, and drives them with `react()`. Needs `GITHUB_TOKEN`. |
