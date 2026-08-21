# pytest: ollama, e2e
"""Example demonstrating Pattern 2 (components in context) using public APIs.

PATTERN 2: Components in Context + Auto Tool Extraction

This example shows the recommended approach for tool calling: add components
to the session context, then use act() with tool_calls=True. The backend
automatically extracts tools via add_tools_from_context_actions().

Key features:
1. Components live in session context with templates
2. Backend auto-extracts tools when tool_calls=True (NO ModelOption.TOOLS needed)
3. Component ID-based prefixing prevents name collisions
4. Tool calls executed via _call_tools() to enable telemetry
5. Multi-turn stability: same components always get same IDs

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
        return f"Database: [{sql}] returned 42 rows"

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
        return f"Search: '{text}' found 5 documents"

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

    def parts(self) -> list[Component | CBlock | ModelOutputThunk]:
        """Return parts of this component."""
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format component for LLM with database query tool."""
        return TemplateRepresentation(
            obj=self,
            args={"description": "Database query interface"},
            tools={"query": MelleaTool.from_callable(QueryDatabaseTool().run)},
            template="🗄️  **Database**: {{description}}\nAvailable: SQL query tool",
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the LLM response."""
        return str(computed.value)


class SearchComponent(Component):
    """Component that provides search capabilities."""

    description = "Search interface"

    def parts(self) -> list[Component | CBlock | ModelOutputThunk]:
        """Return parts of this component."""
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format component for LLM with search tool."""
        return TemplateRepresentation(
            obj=self,
            args={"description": "Search interface"},
            tools={"query": MelleaTool.from_callable(SearchIndexTool().run)},
            template="🔍 **Search**: {{description}}\nAvailable: Document search tool",
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the LLM response."""
        return str(computed.value)


class QueryComponent(Component):
    """Component that prompts the LLM to use available tools."""

    def parts(self) -> list[Component | CBlock | ModelOutputThunk]:
        """Return parts of this component."""
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format component for LLM with explicit tool use instructions."""
        return (
            "Please complete these tasks using the available tools:\n"
            "1. Query the database: SELECT * FROM users WHERE country = 'USA'\n"
            "2. Search documentation: user management best practices\n"
            "Use both the database query tool and the search tool."
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the LLM response."""
        return str(computed.value)


def main() -> None:
    """Demonstrate Pattern 2: Components in context with auto tool extraction."""

    print("\n" + "=" * 70)
    print("PATTERN 2: Components in Context + Auto Tool Extraction")
    print("=" * 70)

    backend = OpenAIBackend(
        model_id=IBM_GRANITE_4_HYBRID_MICRO.ollama_name,  # type: ignore[arg-type]
        formatter=TemplateFormatter(
            model_id=IBM_GRANITE_4_HYBRID_MICRO.hf_model_name  # type: ignore[arg-type]
        ),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    )
    # Use ChatContext (required for context-based tool extraction)
    session = MelleaSession(backend=backend, ctx=ChatContext())

    print("\nStep 1: Add components to session context")
    db_component = DatabaseComponent()
    search_component = SearchComponent()
    query_component = QueryComponent()

    session.ctx = (
        session.ctx.add(db_component).add(search_component).add(query_component)
    )
    print("  ✓ Added DatabaseComponent (has 'query' tool)")
    print("  ✓ Added SearchComponent (also has 'query' tool)")
    print("  ✓ Added QueryComponent (provides context)")

    print("\nStep 2: Use act() with tool_calls=True")
    print("  Configuration:")
    print("  - ModelOption.TOOLS: NO (not needed!)")
    print("  - tool_calls=True: YES (enables auto-extraction)")
    print("  Backend will:")
    print("  - Auto-extract tools from context components")
    print("  - Prefix duplicate names with component IDs")
    print("  - Generate tool calls if LLM requests them")
    print("  Then execute tools via _call_tools() to record telemetry")

    # Use the public API: act() with tool_calls=True
    # NO ModelOption.TOOLS - backend auto-extracts from context!
    # strategy=None to avoid sampling (which may suppress tool calls)
    response = session.act(
        query_component,
        strategy=None,
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

    print("=" * 70)
    print("✓ Pattern 2 (components in context) successfully demonstrated")
    print("✓ Tools automatically extracted from context")
    print("✓ Tool calls executed via _call_tools() for telemetry")
    print("=" * 70)

    print("\nKey Concepts:")
    print("  - Pattern 1: Extract tools only (simple tool calling)")
    print(
        "  - Pattern 2: Components in context with auto-extraction (implicit tool passing)"
    )
    print("  - Both patterns use component ID-based prefixing")
    print("  - Use act() with tool_calls=True (recommended public API)")
    print("  - Tool execution and telemetry handled automatically")


if __name__ == "__main__":
    main()
