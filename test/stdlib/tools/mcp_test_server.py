# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal MCP server for the transport tests in `test_mcp_transports.py`.

Runs on both mcp 1.x (`FastMCP`) and mcp 2.x (`MCPServer`).

Usage:
    python mcp_test_server.py stdio
    python mcp_test_server.py streamable_http <port>
    python mcp_test_server.py sse <port>
"""

import importlib
import sys
from typing import Any


def _load() -> tuple[Any, Any]:
    """Return (server class, ToolError) for whichever mcp major is installed."""
    try:
        server_mod = importlib.import_module("mcp.server.mcpserver")
        errors_mod = importlib.import_module("mcp.server.mcpserver.exceptions")
        return server_mod.MCPServer, errors_mod.ToolError
    except ImportError:  # mcp 1.x
        server_mod = importlib.import_module("mcp.server.fastmcp")
        errors_mod = importlib.import_module("mcp.server.fastmcp.exceptions")
        return server_mod.FastMCP, errors_mod.ToolError


MCPServer, ToolError = _load()

server = MCPServer("mellea-test-server")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def explode() -> str:
    """Always fail."""
    raise ToolError("boom")


if __name__ == "__main__":
    transport = sys.argv[1]
    if transport == "stdio":
        server.run("stdio")
    else:
        import uvicorn

        app = (
            server.streamable_http_app()
            if transport == "streamable_http"
            else server.sse_app()
        )
        uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
