"""Tests for MCP tool discovery and MelleaTool wrapping."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from mellea.backends.tools import MelleaTool
from mellea.stdlib.tools.mcp import (
    discover_mcp_tools,
    http_connection,
    sse_connection,
    stdio_connection,
)


@pytest.fixture
def connection():
    return http_connection("https://example.com/mcp")


def _make_mcp_tool(name: str, description: str = "", schema: dict | None = None):
    """Build a minimal stand-in for mcp.types.Tool."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _make_session(*tools):
    """Build a mock MCP ClientSession with list_tools and call_tool."""
    session = AsyncMock()
    list_result = SimpleNamespace(tools=list(tools))
    session.list_tools = AsyncMock(return_value=list_result)
    return session


def _mock_open_session(session):
    """Return a patch for _open_session that yields the given mock session."""

    @asynccontextmanager
    async def _fake(*args, **kwargs):
        yield session

    return patch("mellea.stdlib.tools.mcp._open_session", new=_fake)


def _call_result(*blocks, is_error=False):
    return CallToolResult(content=list(blocks), isError=is_error)


class TestAsMelleaTool:
    @pytest.mark.asyncio
    async def test_produces_mellea_tool_with_correct_name(self, connection):
        session = _make_session(_make_mcp_tool("get_me", "Get current user"))
        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        tool = specs[0].as_mellea_tool()
        assert isinstance(tool, MelleaTool)
        assert tool.name == "get_me"

    @pytest.mark.asyncio
    async def test_json_schema_structure(self, connection):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        session = _make_session(_make_mcp_tool("search", "Search", schema))
        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        tool = specs[0].as_mellea_tool()
        fn = tool.as_json_tool["function"]
        assert fn["name"] == "search"
        assert fn["description"] == "Search"
        assert fn["parameters"] == schema


class TestSyncWrapper:
    @pytest.mark.asyncio
    async def test_extracts_text_from_content_blocks(self, connection):
        call_result = _call_result(TextContent(type="text", text="hello world"))
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run(q="test")

        assert output == "hello world"

    @pytest.mark.asyncio
    async def test_joins_multiple_content_blocks(self, connection):
        call_result = _call_result(
            TextContent(type="text", text="line1"),
            TextContent(type="text", text="line2"),
        )
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run()

        assert output == "line1\nline2"

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_string(self, connection):
        call_result = _call_result()
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run()

        assert output == ""

    @pytest.mark.asyncio
    async def test_none_kwargs_stripped(self, connection):
        """Kwargs with None values are not forwarded to call_tool."""
        received: list[dict] = []

        async def _capture(tool_name, *, arguments):
            received.append(arguments)
            return _call_result()

        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = _capture

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            tool.run(q="test", page=None)

        assert received == [{"q": "test"}]

    @pytest.mark.asyncio
    async def test_embedded_resource_text_extracted(self, connection):
        resource = TextResourceContents(uri="file://doc.txt", text="resource text")
        call_result = _call_result(EmbeddedResource(type="resource", resource=resource))
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run()

        assert output == "resource text"

    @pytest.mark.asyncio
    async def test_image_content_returns_binary_descriptor(self, connection):
        call_result = _call_result(
            ImageContent(type="image", data="abc123", mimeType="image/png")
        )
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run()

        assert output == "[binary: image/png]"

    @pytest.mark.asyncio
    async def test_blob_resource_returns_binary_descriptor(self, connection):
        resource = BlobResourceContents(
            uri="file://data.pdf", blob="abc123", mimeType="application/pdf"
        )
        call_result = _call_result(EmbeddedResource(type="resource", resource=resource))
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run()

        assert output == "[binary: application/pdf]"

    @pytest.mark.asyncio
    async def test_audio_content_returns_binary_descriptor(self, connection):
        call_result = _call_result(
            AudioContent(type="audio", data="abc123", mimeType="audio/mpeg")
        )
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run()

        assert output == "[binary: audio/mpeg]"

    @pytest.mark.asyncio
    async def test_resource_link_resolved_to_text(self, connection):
        link = ResourceLink(type="resource_link", uri="file://doc.txt", name="doc")
        call_result = _call_result(link)
        resource_result = SimpleNamespace(
            contents=[TextResourceContents(uri="file://doc.txt", text="linked content")]
        )
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)
        session.read_resource = AsyncMock(return_value=resource_result)

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run()

        assert output == "linked content"

    @pytest.mark.asyncio
    async def test_resource_link_falls_back_on_read_failure(self, connection):
        link = ResourceLink(
            type="resource_link", uri="https://example.com/doc", name="doc"
        )
        call_result = _call_result(link)
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)
        session.read_resource = AsyncMock(side_effect=Exception("not found"))

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run()

        assert output == "[resource: https://example.com/doc]"

    @pytest.mark.asyncio
    async def test_is_error_returns_error_string(self, connection):
        call_result = _call_result(
            TextContent(type="text", text="not found"), is_error=True
        )
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run()

        assert output == "[tool error] not found"

    @pytest.mark.asyncio
    async def test_is_error_no_content_returns_fallback(self, connection):
        call_result = _call_result(is_error=True)
        session = _make_session(_make_mcp_tool("my_tool"))
        session.call_tool = AsyncMock(return_value=call_result)

        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        with _mock_open_session(session):
            tool = specs[0].as_mellea_tool()
            output = tool.run()

        assert output == "[tool error] tool call failed"


class TestDiscoverMcpTools:
    @pytest.mark.asyncio
    async def test_returns_specs_for_each_tool(self, connection):
        session = _make_session(
            _make_mcp_tool("get_me", "Return the current user"),
            _make_mcp_tool("search_pull_requests", "Search PRs"),
        )
        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        assert len(specs) == 2
        assert [s.name for s in specs] == ["get_me", "search_pull_requests"]

    @pytest.mark.asyncio
    async def test_empty_server_returns_empty_list(self, connection):
        session = _make_session()
        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        assert specs == []

    @pytest.mark.asyncio
    async def test_spec_fields_populated(self, connection):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        session = _make_session(_make_mcp_tool("search", "Search things", schema))
        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        s = specs[0]
        assert s.name == "search"
        assert s.description == "Search things"
        assert s.input_schema == schema
        assert s._connection == connection

    @pytest.mark.asyncio
    async def test_none_input_schema_becomes_empty_dict(self, connection):
        tool = SimpleNamespace(name="x", description="", inputSchema=None)
        session = AsyncMock()
        session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[tool]))
        with _mock_open_session(session):
            specs = await discover_mcp_tools(connection)

        assert specs[0].input_schema == {}


class TestHttpConnection:
    def test_sets_transport(self):
        assert http_connection("https://example.com")["transport"] == "streamable_http"

    def test_sets_url(self):
        assert http_connection("https://example.com")["url"] == "https://example.com"

    def test_token_sets_auth_header(self):
        conn = http_connection("https://example.com", api_key="abc")
        assert conn["headers"]["Authorization"] == "Bearer abc"

    def test_no_token_empty_headers(self):
        assert http_connection("https://example.com")["headers"] == {}

    def test_extra_headers_merged(self):
        conn = http_connection(
            "https://example.com", api_key="abc", headers={"X-Custom": "val"}
        )
        assert conn["headers"]["Authorization"] == "Bearer abc"
        assert conn["headers"]["X-Custom"] == "val"

    def test_headers_only(self):
        conn = http_connection("https://example.com", headers={"X-Key": "v"})
        assert conn["headers"] == {"X-Key": "v"}


class TestSseConnection:
    def test_sets_transport(self):
        assert sse_connection("https://example.com")["transport"] == "sse"

    def test_token_sets_auth_header(self):
        conn = sse_connection("https://example.com", api_key="tok")
        assert conn["headers"]["Authorization"] == "Bearer tok"


class TestStdioConnection:
    def test_sets_transport(self):
        assert stdio_connection("gh")["transport"] == "stdio"

    def test_sets_command(self):
        assert stdio_connection("gh")["command"] == "gh"

    def test_args_included_when_provided(self):
        conn = stdio_connection("gh", args=["mcp", "serve"])
        assert conn["args"] == ["mcp", "serve"]

    def test_args_omitted_when_not_provided(self):
        assert "args" not in stdio_connection("gh")

    def test_env_included_when_provided(self):
        conn = stdio_connection("gh", env={"TOKEN": "x"})
        assert conn["env"] == {"TOKEN": "x"}

    def test_env_omitted_when_not_provided(self):
        assert "env" not in stdio_connection("gh")
