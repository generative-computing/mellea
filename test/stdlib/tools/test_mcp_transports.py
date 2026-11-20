# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport tests for `mellea.stdlib.tools.mcp` against a real local MCP server.

`test_mcp.py` mocks `_open_session`, so the transport code (HTTP client
construction, stream unpacking, session setup) never runs there. These tests
start `mcp_test_server.py` in a subprocess and talk to it over stdio,
Streamable HTTP and SSE. They run unchanged on mcp 1.x and mcp 2.x.
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mellea.stdlib.tools.mcp import (
    discover_mcp_tools,
    http_connection,
    sse_connection,
    stdio_connection,
)

pytestmark = pytest.mark.integration

_SERVER = str(Path(__file__).with_name("mcp_test_server.py"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"test MCP server exited early ({proc.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("test MCP server did not start listening")


@pytest.fixture(params=["stdio", "streamable_http", "sse"])
def connection(request):
    transport = request.param
    if transport == "stdio":
        yield stdio_connection(sys.executable, args=[_SERVER, "stdio"], timeout=30)
        return

    port = _free_port()
    proc = subprocess.Popen([sys.executable, _SERVER, transport, str(port)])
    try:
        _wait_for_port(port, proc)
        if transport == "streamable_http":
            yield http_connection(f"http://127.0.0.1:{port}/mcp", read_timeout=30)
        else:
            yield sse_connection(f"http://127.0.0.1:{port}/sse", read_timeout=30)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


async def test_discover_lists_tools_with_schema(connection):
    specs = {s.name: s for s in await discover_mcp_tools(connection)}

    assert set(specs) == {"add", "explode"}
    assert specs["add"].description == "Add two integers."
    props = specs["add"].input_schema["properties"]
    assert set(props) == {"a", "b"}


async def test_tool_call_returns_text(connection):
    specs = {s.name: s for s in await discover_mcp_tools(connection)}

    assert specs["add"].as_mellea_tool().run(a=2, b=3) == "5"


async def test_tool_error_is_reported_not_raised(connection):
    specs = {s.name: s for s in await discover_mcp_tools(connection)}

    result = specs["explode"].as_mellea_tool().run()

    assert result.startswith("[tool error]")
    assert "boom" in result
