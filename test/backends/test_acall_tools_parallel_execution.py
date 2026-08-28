# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test for acall_tools() execution with parallel same-name tool calls.

This test fills the gap identified in PR #1431 review: verifying that the
automatic tool execution loop through acall_tools() correctly processes
multiple same-name tool calls from the list-based tool_calls structure.

Coverage layers:
- Extraction layer: test/helpers/test_openai_compatible_helpers.py::test_duplicate_same_name_tool_calls
  Verifies that extract_model_tool_requests() preserves both calls in the list.
- Execution layer: this file
  Verifies that acall_tools() iterates and executes all calls from the list,
  producing ToolMessages for each (not just the last one).
"""

import pytest

from mellea.backends.tools import MelleaTool
from mellea.core.base import ModelOutputThunk, ModelToolCall
from mellea.stdlib.functional import acall_tools

pytestmark = [pytest.mark.integration]


@pytest.fixture
def backend(mock_ollama_backend):
    """Create an OllamaModelBackend for formatter.print() only.

    Note: acall_tools() only uses backend.formatter, not inference.
    Tests use local Python functions as tool implementations, no model calls.
    """
    return mock_ollama_backend()


@pytest.mark.asyncio
async def test_acall_tools_executes_all_parallel_same_name_calls(backend):
    """Verify acall_tools() executes all parallel same-name tool calls.

    This is the execution-level regression test for PR #1431.
    It directly tests acall_tools() to ensure:
    1. All tool calls in the list are iterated (not lost to dict key collision)
    2. Each produces a ToolMessage
    3. The returned list has correct cardinality

    With dict-based tool_calls, only the last call would execute.
    With list-based tool_calls, all calls must execute.
    """
    execution_log = []

    def search(query: str) -> str:
        """A search tool that records executions."""
        execution_log.append(query)
        return f"Results for: {query}"

    # Create mock tool calls (simulating model's response)
    tool = MelleaTool.from_callable(search)
    tool_call_1 = ModelToolCall(
        name="search",
        func=tool,
        args={"query": "Python programming"},
        tool_call_id="call_1",
    )
    tool_call_2 = ModelToolCall(
        name="search",
        func=tool,
        args={"query": "JavaScript frameworks"},
        tool_call_id="call_2",
    )

    # Create ModelOutputThunk with list of tool calls
    mot = ModelOutputThunk(
        value="I'll search for both topics.", tool_calls=[tool_call_1, tool_call_2]
    )

    # Call acall_tools() - the automatic execution pipeline
    tool_messages = await acall_tools(mot, backend)

    # Verify all tool calls were executed
    assert len(execution_log) == 2, (
        f"Both tool calls should execute via acall_tools(), "
        f"got {len(execution_log)} executions"
    )

    # Verify distinct execution
    assert "Python programming" in execution_log
    assert "JavaScript frameworks" in execution_log

    # Verify ToolMessages produced
    assert len(tool_messages) == 2, (
        f"Should produce 2 ToolMessages, got {len(tool_messages)}"
    )
    for msg in tool_messages:
        assert msg.role == "tool"
        assert msg.content  # Each has content from execution


@pytest.mark.asyncio
async def test_acall_tools_preserves_order_in_execution(backend):
    """Verify acall_tools() executes tool calls in order.

    Order preservation is critical for reproducibility and correctness,
    especially when tool results depend on prior execution (e.g., write then read).
    """
    execution_order = []

    def log_operation(operation: str, index: int) -> str:
        """Records operation and index to track execution order."""
        execution_order.append((operation, index))
        return f"Executed {operation} #{index}"

    tool = MelleaTool.from_callable(log_operation)

    # Create tool calls in specific order
    tool_calls = [
        ModelToolCall(
            name="log_operation",
            func=tool,
            args={"operation": "write", "index": 1},
            tool_call_id="call_1",
        ),
        ModelToolCall(
            name="log_operation",
            func=tool,
            args={"operation": "read", "index": 2},
            tool_call_id="call_2",
        ),
        ModelToolCall(
            name="log_operation",
            func=tool,
            args={"operation": "validate", "index": 3},
            tool_call_id="call_3",
        ),
    ]

    mot = ModelOutputThunk(value="Running three operations.", tool_calls=tool_calls)

    # Execute through acall_tools()
    tool_messages = await acall_tools(mot, backend)

    # Verify execution happened in order
    assert len(execution_order) == 3
    assert execution_order[0] == ("write", 1)
    assert execution_order[1] == ("read", 2)
    assert execution_order[2] == ("validate", 3)

    # Verify ToolMessages match order
    assert len(tool_messages) == 3
    for i, msg in enumerate(tool_messages):
        assert msg.role == "tool"
        # Each message should reflect the execution output
        assert f"#{i + 1}" in msg.content


@pytest.mark.asyncio
async def test_acall_tools_with_mixed_tools(backend):
    """Verify acall_tools() handles multiple different tools alongside duplicates.

    Realistic scenario: user calls search twice, calculate once, search again.
    tool_calls should be: [search, search, calculate, search]
    """
    executions = []

    def search(query: str) -> str:
        executions.append(("search", query))
        return f"Search: {query}"

    def calculate(expr: str) -> str:
        executions.append(("calculate", expr))
        return f"Calc: {expr}"

    search_tool = MelleaTool.from_callable(search)
    calc_tool = MelleaTool.from_callable(calculate)

    tool_calls = [
        ModelToolCall(
            name="search", func=search_tool, args={"query": "AI"}, tool_call_id="call_1"
        ),
        ModelToolCall(
            name="search", func=search_tool, args={"query": "ML"}, tool_call_id="call_2"
        ),
        ModelToolCall(
            name="calculate",
            func=calc_tool,
            args={"expr": "2+2"},
            tool_call_id="call_3",
        ),
        ModelToolCall(
            name="search", func=search_tool, args={"query": "DL"}, tool_call_id="call_4"
        ),
    ]

    mot = ModelOutputThunk(value="Complex query", tool_calls=tool_calls)

    tool_messages = await acall_tools(mot, backend)

    # Verify all 4 executions happened
    assert len(executions) == 4
    assert len(tool_messages) == 4

    # Verify order and content
    assert executions[0] == ("search", "AI")
    assert executions[1] == ("search", "ML")
    assert executions[2] == ("calculate", "2+2")
    assert executions[3] == ("search", "DL")


@pytest.mark.asyncio
async def test_acall_tools_cardinality_regression(backend):
    """Regression test: verify tool_calls cardinality == ToolMessages cardinality.

    With dict-based tool_calls:
    - {search: call_1, search: call_2} → only last key kept
    - Returns 1 ToolMessage (dict overwrite)

    With list-based tool_calls:
    - [call_1(search), call_2(search)] → both kept
    - Returns 2 ToolMessages (list iteration)

    This assertion catches any regression back to dict-based behavior.
    """
    executions = []

    def dummy_tool(x: str) -> str:
        executions.append(x)
        return f"Result: {x}"

    tool = MelleaTool.from_callable(dummy_tool)

    # Create N identical tool calls with different args
    n_calls = 5
    tool_calls = [
        ModelToolCall(
            name="dummy_tool",
            func=tool,
            args={"x": f"input_{i}"},
            tool_call_id=f"call_{i}",
        )
        for i in range(n_calls)
    ]

    mot = ModelOutputThunk(value="Run N times", tool_calls=tool_calls)

    tool_messages = await acall_tools(mot, backend)

    # THE CRITICAL ASSERTION: cardinality must match
    assert len(tool_messages) == n_calls, (
        f"Cardinality regression: expected {n_calls} ToolMessages, "
        f"got {len(tool_messages)}. Dict-based tool_calls would collapse to 1."
    )
    assert len(executions) == n_calls
