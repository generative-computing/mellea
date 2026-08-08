# pytest: ollama, e2e
"""Example demonstrating Pattern 2: Components in context (using private APIs).

⚠️  DEPRECATED: This example uses private _call_tools() API. For new code, prefer
`pattern2_public_api.py` which uses the public MelleaSession.act() API.

Pattern 2 combines both approaches:
1. Add components to session context (for rendering in the prompt)
2. Extract and explicitly pass tools via ModelOption.TOOLS (for tool calling)

This allows the LLM to see the components in the conversation AND use their tools.

To view tool calling telemetry metrics:
    export MELLEA_METRICS_ENABLED=true
    export MELLEA_METRICS_CONSOLE=true
    uv run python this_script.py

Tool calls are executed via Mellea's pipeline to enable telemetry recording.
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


def main():
    """Demonstrate Pattern 2: Components in context + auto tool extraction."""
    print("=" * 70)
    print("PATTERN 2: Components in Context + Auto Tool Extraction")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("STEP 1: Create Components")
    print("=" * 70)

    # Create components
    db_component = DatabaseComponent()
    search_component = SearchComponent()
    print("✓ Created DatabaseComponent and SearchComponent with tools")

    print("\n" + "=" * 70)
    print("STEP 2: Add Components to Context")
    print("=" * 70)

    # Add components to context (no templates needed - already defined!)
    print("\nAdding components to context...")
    db_component = DatabaseComponent()
    search_component = SearchComponent()
    session_ctx = ChatContext()
    session_ctx = session_ctx.add(db_component)
    session_ctx = session_ctx.add(search_component)
    print("  ✓ Added both components to context")

    print("\n" + "=" * 70)
    print("STEP 3: Set Up Backend and Session")
    print("=" * 70)

    ollama_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
    if not ollama_host.startswith(("http://", "https://")):
        ollama_host = f"http://{ollama_host}"

    backend = OpenAIBackend(
        model_id=IBM_GRANITE_4_HYBRID_MICRO.ollama_name,  # type: ignore[arg-type]
        formatter=TemplateFormatter(
            model_id=IBM_GRANITE_4_HYBRID_MICRO.hf_model_name  # type: ignore[arg-type]
        ),
        base_url=f"{ollama_host}/v1",
        api_key="ollama",
    )
    session = MelleaSession(backend, ctx=session_ctx)

    print("\nSession created with components in context")

    print("\n" + "=" * 70)
    print("STEP 4: LLM Generation with Tool Calling")
    print("=" * 70)

    prompt = (
        "I need to find information about users. Please:"
        "\n1. Query the database to get all users"
        "\n2. Search the documentation for user management best practices"
        "\nUse the available tools to complete both tasks."
    )

    print(f"\nPrompt:\n{prompt}")

    # IMPORTANT: NO ModelOption.TOOLS - backend auto-extracts from context!
    print("\nCalling session.instruct() with:")
    print("  - Components in context: YES (database + search)")
    print("  - ModelOption.TOOLS: NO (backend auto-extracts!)")
    print("  - tool_calls: True")

    response = session.instruct(
        prompt,
        model_options={
            ModelOption.TOOL_CHOICE: "auto",
            ModelOption.MAX_NEW_TOKENS: 1000,
        },
        strategy=None,
        tool_calls=True,
    )

    print(f"\nLLM Response:\n{response.value}\n")

    # Execute tool calls
    if response.tool_calls:
        print(f"Tool calls requested by LLM: {list(response.tool_calls.keys())}")
        print("\nExecuting tool calls via Mellea's pipeline (hooks enabled):")
        # Execute tools through Mellea's pipeline to trigger telemetry hooks
        tool_messages = _call_tools(response, backend)
        for msg in tool_messages:
            print(f"  {msg.name}() → {msg.content}")
        print("\nNote: Tool calls are executed via _call_tools() so telemetry")
        print("hooks fire and metrics are recorded in mellea.tool.calls")
        print("\n✓ Tool calling SUCCESSFUL in Pattern 2")
    else:
        print("No tool calls were requested by the LLM.")

    print("\n" + "=" * 70)
    print("Pattern 2 Summary")
    print("=" * 70)
    print("""
PATTERN 2: Components in Context + Auto Tool Extraction

Approach:
  1. Create components with tools and templates
  2. Create session with ChatContext
  3. Add components to context: session.ctx = session.ctx.add(component)
  4. Call session.instruct() with tool_calls=True (NO ModelOption.TOOLS!)

Benefits:
  ✓ Components rendered in the prompt
  ✓ Components' tools automatically extracted and available
  ✓ ID-based prefixing prevents tool collisions
  ✓ Multi-turn stable (same instances = same IDs)
  ✓ No explicit tool passing needed

Key Point:
  The backend automatically calls add_tools_from_context_actions() when
  tool_calls=True, extracting tools from all components in the context.
  This makes Pattern 2 truly implicit and elegant.

When to Use Pattern 2:
  - You need components to appear in the conversation
  - You want their tools available for calling automatically
  - You prefer implicit over explicit tool passing
    """)

    print("=" * 70)
    print("✓ Pattern 2 Successfully Demonstrated")
    print("=" * 70)


if __name__ == "__main__":
    main()
