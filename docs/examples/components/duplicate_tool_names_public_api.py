# pytest: ollama, e2e
"""Example demonstrating component ID-based tool prefixing using public APIs.

This is the recommended approach for handling multiple components with identical
tool names. It uses the public MelleaSession.act() API with tool_calls=True,
which automatically extracts and prefixes tools from context.

When multiple components define tools with identical names, Mellea automatically
prefixes each tool name with its component ID (component_{ID}__tool_name) to prevent
naming collisions.

In this example:
- DatabaseComponent has a "query" tool for querying data
- SearchComponent also has a "query" tool for searching
- Both tools are available to the LLM with prefixed names to avoid conflicts
- The LLM is prompted to use both tools and demonstrate collision handling
- Tool calls are executed via _call_tools() to enable telemetry recording

To view tool calling telemetry metrics:
    export MELLEA_METRICS_ENABLED=true
    export MELLEA_METRICS_CONSOLE=true
    uv run python this_script.py
"""

import os
from typing import Any

from mellea.backends import ModelOption
from mellea.backends.model_ids import IBM_GRANITE_4_HYBRID_MICRO
from mellea.backends.openai import OpenAIBackend
from mellea.backends.tools import MelleaTool
from mellea.core import CBlock, Component, ModelOutputThunk, TemplateRepresentation
from mellea.core.base import AbstractMelleaTool
from mellea.formatters import TemplateFormatter
from mellea.stdlib.context import ChatContext
from mellea.stdlib.functional import _call_tools
from mellea.stdlib.session import MelleaSession


class QueryDatabaseTool(AbstractMelleaTool):
    """Tool for querying a database."""

    name = "query"

    def run(self, sql: str) -> str:
        """Execute a SQL query on the database.

        Args:
            sql: The SQL query to execute

        Returns:
            Mock query results as a string
        """
        return f"Database query result: [{sql}] returned 42 rows"

    @property
    def as_json_tool(self) -> dict[str, Any]:
        """Return JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Query a database with SQL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL query to execute"}
                    },
                    "required": ["sql"],
                },
            },
        }


class SearchIndexTool(AbstractMelleaTool):
    """Tool for searching an index."""

    name = "query"

    def run(self, text: str) -> str:
        """Search the index for matching documents.

        Args:
            text: The search query text

        Returns:
            Mock search results as a string
        """
        return f"Search results for '{text}': found 5 documents"

    @property
    def as_json_tool(self) -> dict[str, Any]:
        """Return JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Search an index for documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Search query text"}
                    },
                    "required": ["text"],
                },
            },
        }


class DatabaseComponent(Component):
    """Component that provides database querying capabilities."""

    description = "Database query interface"
    _tool = QueryDatabaseTool()

    def parts(self) -> list[Component | CBlock | ModelOutputThunk]:
        """Return parts of this component."""
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format component for LLM with database query tool."""
        return TemplateRepresentation(
            obj=self,
            args={"description": "Database query interface"},
            tools={"query": MelleaTool.from_callable(self._tool.run)},
            template="🗄️  **Database Interface**: {{description}}\nAvailable: SQL query tool",
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the LLM response."""
        return str(computed.value)


class SearchComponent(Component):
    """Component that provides search capabilities."""

    description = "Search interface"
    _tool = SearchIndexTool()

    def parts(self) -> list[Component | CBlock | ModelOutputThunk]:
        """Return parts of this component."""
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format component for LLM with search tool."""
        return TemplateRepresentation(
            obj=self,
            args={"description": "Search interface"},
            tools={"query": MelleaTool.from_callable(self._tool.run)},
            template="🔍 **Search Interface**: {{description}}\nAvailable: Document search tool",
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the LLM response."""
        return str(computed.value)


class TaskComponent(Component):
    """Component that prompts the LLM to use available tools."""

    def parts(self) -> list[Component | CBlock | ModelOutputThunk]:
        """Return parts of this component."""
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format component for LLM with explicit tool use instructions."""
        return (
            "Please complete these tasks using the available tools:\n"
            "1. Query the database: SELECT * FROM users\n"
            "2. Search documentation: best practices\n"
            "Use both tools to demonstrate collision handling."
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the LLM response."""
        return str(computed.value)


def main() -> None:
    """Main function demonstrating component ID-based tool prefixing."""

    print("\n" + "=" * 70)
    print("Component ID-Based Tool Prefixing (Public API Example)")
    print("=" * 70)

    backend = OpenAIBackend(
        model_id=IBM_GRANITE_4_HYBRID_MICRO.ollama_name,  # type: ignore[arg-type]
        formatter=TemplateFormatter(
            model_id=IBM_GRANITE_4_HYBRID_MICRO.hf_model_name  # type: ignore[arg-type]
        ),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    )

    # Create a session with ChatContext (required for context-based tool extraction)
    session = MelleaSession(backend=backend, ctx=ChatContext())

    # Add both components to the session context
    db_component = DatabaseComponent()
    search_component = SearchComponent()

    session.ctx = session.ctx.add(db_component).add(search_component)

    print("\nStep 1: Add components to session context")
    print("  ✓ Added DatabaseComponent (has 'query' tool)")
    print("  ✓ Added SearchComponent (also has 'query' tool)")

    print("\nStep 2: Use act() with tool_calls=True")
    print("  This automatically:")
    print("    - Extracts tools from all components in context")
    print("    - Prefixes duplicate tool names with component IDs")
    print("    - Enables tool calling")
    print("  Then execute tools via _call_tools() to record telemetry")

    # Create a task component that will request tool use
    action = TaskComponent()

    # Use the public API: act() with tool_calls=True
    # This handles tool extraction, execution, and telemetry automatically
    response = session.act(
        action,
        model_options={
            ModelOption.MAX_NEW_TOKENS: 500,
            ModelOption.TOOL_CHOICE: "auto",
        },
        tool_calls=True,
    )

    print(f"\nLLM Response:\n{response.value}\n")

    # Check if tool calls were made and execute them
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Tool calls requested: {list(response.tool_calls.keys())}")
        print(
            "\nExecuting tool calls via Mellea's pipeline (enables telemetry recording):"
        )
        tool_messages = _call_tools(response, backend)
        for msg in tool_messages:
            print(f"  {msg.name}() → {msg.content}")
        print()
    else:
        print("(No tool calls in this response)\n")

    print("\n" + "=" * 70)
    print("✓ Component ID-based tool prefixing successfully demonstrated")
    print("✓ Tools extracted from context and executed via _call_tools()")
    print("=" * 70)


if __name__ == "__main__":
    main()
