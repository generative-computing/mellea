"""MCP tool discovery and MelleaTool wrapping.

Bridges [MCP](https://modelcontextprotocol.io/) server tools into Mellea's
native tool-calling system. Connect to any MCP server and use its tools directly
inside a Mellea agent.

Two-step workflow:

1. ``discover_mcp_tools`` — inspect a server's tools without instantiating them.
2. ``MCPToolSpec.as_mellea_tool`` — build a callable ``MelleaTool`` from a spec.

Connection helpers (``http_connection``, ``sse_connection``, ``stdio_connection``)
produce the config dicts fed into the functions above.

Each tool invocation opens its own short-lived MCP session, so no session
lifetime management is required by the caller. Async MCP calls are executed on
Mellea's shared background event loop via ``_run_async_in_thread``.
"""

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

try:
    import httpx
    from mcp import ClientSession, StdioServerParameters, stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.types import (
        AudioContent,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
        TextResourceContents,
    )
except ImportError as e:
    raise ImportError(
        "MCP integration requires the `mcp` and `httpx` packages. "
        "Please install mellea with tools support: pip install 'mellea[tools]'"
    ) from e

from mellea.backends.tools import MelleaTool
from mellea.helpers.event_loop_helper import _run_async_in_thread


class MCPToolSpec:
    """Metadata for a single tool from an MCP server.

    Holds everything needed to inspect or instantiate a ``MelleaTool`` without
    keeping a live session open.

    Args:
        name (str): Tool name as registered on the server.
        description (str): Human-readable description from the server.
        input_schema (dict[str, Any]): OpenAI-compatible parameters schema dict.
        connection (dict[str, Any]): Transport config dict returned by one of the
            connection helpers (``http_connection``, ``sse_connection``,
            ``stdio_connection``).
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        connection: dict[str, Any],
    ) -> None:
        """Store the spec fields and the transport config."""
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._connection = connection

    def as_mellea_tool(self) -> MelleaTool:
        """Create a callable ``MelleaTool`` from this spec.

        The returned tool opens a fresh MCP session per call. For ``stdio``
        transport this means a new subprocess is spawned on every tool
        invocation; prefer ``streamable_http`` or ``sse`` for
        performance-sensitive use.

        Returns:
            A ``MelleaTool`` instance ready to pass via ``ModelOption.TOOLS``
            or to an agent loop like ``react()``.
        """
        return MelleaTool(
            self.name,
            _make_sync_call(self._connection, self.name),
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.input_schema,
                },
            },
        )

    def __repr__(self) -> str:
        """Short debug-friendly representation."""
        return f"MCPToolSpec(name={self.name!r})"


async def discover_mcp_tools(connection: dict[str, Any]) -> list[MCPToolSpec]:
    """Discover all tools on an MCP server and return their metadata.

    Opens a single session, calls ``list_tools()``, then closes. No
    ``MelleaTool`` objects are instantiated — callers can inspect and filter
    the returned specs before calling ``MCPToolSpec.as_mellea_tool``.

    Args:
        connection: Transport config dict. Build it with one of the connection
            helpers rather than constructing it by hand: ``http_connection``,
            ``sse_connection``, ``stdio_connection``.

    Returns:
        List of ``MCPToolSpec`` objects, one per tool on the server.
    """
    async with _open_session(connection) as session:
        result = await session.list_tools()
        return [
            MCPToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if tool.inputSchema is not None else {},
                connection=connection,
            )
            for tool in result.tools
        ]


def http_connection(
    url: str,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    connect_timeout: float = 30.0,
    read_timeout: float = 300.0,
) -> dict[str, Any]:
    """Build a Streamable HTTP connection config.

    Args:
        url: MCP server URL.
        api_key: Sets ``Authorization: Bearer <api_key>``.
        headers: Additional headers, merged after ``api_key``.
        connect_timeout: Seconds to wait for TCP connection (default 30).
        read_timeout: Seconds to wait for a response (default 300).

    Returns:
        Connection dict ready to pass to ``discover_mcp_tools``.
    """
    h: dict[str, str] = {}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    if headers:
        h.update(headers)
    return {
        "transport": "streamable_http",
        "url": url,
        "headers": h,
        "connect_timeout": connect_timeout,
        "read_timeout": read_timeout,
    }


def sse_connection(
    url: str,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    connect_timeout: float = 30.0,
    read_timeout: float = 300.0,
) -> dict[str, Any]:
    """Build an SSE connection config.

    Args:
        url: MCP server URL.
        api_key: Sets ``Authorization: Bearer <api_key>``.
        headers: Additional headers, merged after ``api_key``.
        connect_timeout: Seconds to wait for TCP connection (default 30).
        read_timeout: Seconds to wait for a response (default 300).

    Returns:
        Connection dict ready to pass to ``discover_mcp_tools``.
    """
    h: dict[str, str] = {}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    if headers:
        h.update(headers)
    return {
        "transport": "sse",
        "url": url,
        "headers": h,
        "connect_timeout": connect_timeout,
        "read_timeout": read_timeout,
    }


def stdio_connection(
    command: str,
    *,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Build a stdio connection config.

    Args:
        command: Executable to run (e.g. ``"gh"``).
        args: Command-line arguments (e.g. ``["mcp", "serve"]``).
        env: Environment variables for the subprocess.
        timeout: Total seconds allowed for a tool call to complete (default 300).

    Returns:
        Connection dict ready to pass to ``discover_mcp_tools``.
    """
    conn: dict[str, Any] = {
        "transport": "stdio",
        "command": command,
        "read_timeout": timeout,
    }
    if args:
        conn["args"] = args
    if env:
        conn["env"] = env
    return conn


@asynccontextmanager
async def _open_session(connection: dict[str, Any]):
    """Open a fresh MCP ClientSession for the given connection config."""
    transport = connection.get("transport", "streamable_http")

    connect_timeout: float = connection.get("connect_timeout", 30.0)
    read_timeout: float = connection.get("read_timeout", 300.0)

    if transport == "streamable_http":
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )
        async with httpx.AsyncClient(
            headers=connection.get("headers", {}), timeout=timeout
        ) as http_client:
            async with streamable_http_client(
                connection["url"], http_client=http_client
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    elif transport == "sse":
        async with sse_client(
            url=connection["url"],
            headers=connection.get("headers", {}),
            timeout=connect_timeout,
            sse_read_timeout=read_timeout,
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    elif transport == "stdio":
        params = StdioServerParameters(
            command=connection["command"],
            args=connection.get("args", []),
            env=connection.get("env"),
        )
        async with asyncio.timeout(read_timeout):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    else:
        raise ValueError(f"Unknown MCP transport: {transport!r}")


async def _execute_tool(
    connection: dict[str, Any], tool_name: str, kwargs: dict[str, Any]
) -> str:
    """Open a session, invoke the tool, and flatten the result to a string."""
    async with _open_session(connection) as session:
        result = await session.call_tool(tool_name, arguments=kwargs)
        if result.isError:
            error_parts = [
                b.text for b in (result.content or []) if isinstance(b, TextContent)
            ]
            error_msg = "\n".join(error_parts) if error_parts else "tool call failed"
            return f"[tool error] {error_msg}"
        if result.content:
            parts = []
            for block in result.content:
                if isinstance(block, TextContent):
                    parts.append(block.text)
                elif isinstance(block, EmbeddedResource) and isinstance(
                    block.resource, TextResourceContents
                ):
                    parts.append(block.resource.text)
                elif isinstance(block, (ImageContent, AudioContent)):
                    parts.append(f"[binary: {block.mimeType}]")
                elif isinstance(block, ResourceLink):
                    try:
                        resource_result = await session.read_resource(block.uri)
                        for item in resource_result.contents:
                            if isinstance(item, TextResourceContents):
                                parts.append(item.text)
                            else:
                                mime = item.mimeType or "unknown"
                                parts.append(f"[binary: {mime}]")
                    except Exception:
                        parts.append(f"[resource: {block.uri}]")
                elif isinstance(block, EmbeddedResource):
                    # BlobResourceContents
                    mime = block.resource.mimeType or "unknown"
                    parts.append(f"[binary: {mime}]")
            return "\n".join(parts) if parts else ""
        return ""


def _make_sync_call(connection: dict[str, Any], tool_name: str) -> Callable[..., str]:
    """Build a sync wrapper around an async MCP tool invocation.

    Runs the async call on Mellea's shared background event loop via
    ``_run_async_in_thread``. MCP servers expect absent fields rather than
    explicit ``null`` values, so ``None`` kwargs are stripped before the call.
    """

    def sync_call(**kwargs: Any) -> str:
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return _run_async_in_thread(_execute_tool(connection, tool_name, clean))

    return sync_call


__all__ = [
    "MCPToolSpec",
    "discover_mcp_tools",
    "http_connection",
    "sse_connection",
    "stdio_connection",
]
