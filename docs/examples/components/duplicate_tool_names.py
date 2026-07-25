# pytest: ollama, e2e
"""Example demonstrating two components with tools that have the same name.

When multiple components define tools with identical names, Mellea automatically
prefixes each tool name with its component ID (component_{ID}.tool_name) to prevent
naming collisions. This allows safe composition of multiple components.

In this example:
- DatabaseComponent has a "query" tool for querying data
- SearchComponent also has a "query" tool for searching
- Both tools are available to the LLM with prefixed names to avoid conflicts
- The LLM is prompted to use both tools and demonstrate collision handling

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
from mellea.backends.tools import MelleaTool, add_tools_from_context_actions
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
    """Demonstrate two components with the same tool name."""
    # Create two components with tools that have the same name
    db_component = DatabaseComponent()
    search_component = SearchComponent()

    print("=" * 70)
    print("PART 1: Tool Extraction and Prefixing")
    print("=" * 70)

    # Extract tools from components - they will have prefixed names to avoid collisions
    ctx_actions = [db_component, search_component]
    tools = {}
    add_tools_from_context_actions(tools, ctx_actions)

    # Print available tools to show the prefixing in action
    print("\nAvailable tools with component ID-based prefixes:")
    query_tools = []
    for tool_name in sorted(tools.keys()):
        if tool_name.startswith("component_"):
            print(f"  - {tool_name}")
            if "query" in tool_name:
                query_tools.append(tool_name)

    # Both "query" tools are now available with different prefixes
    print(f"\nTotal query tools available: {len(query_tools)}")
    assert len(query_tools) == 2, (
        f"Expected 2 query tools with different component IDs, got {len(query_tools)}"
    )

    print("\n" + "=" * 70)
    print("PART 2: LLM Generation with Tool Calls")
    print("=" * 70)

    # Set up backend and session
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
    # Create a session to call the LLM with tool definitions
    session = MelleaSession(backend, ctx=ChatContext())

    # Generate response with tool calls
    prompt = (
        "I need information about users. First, query the database for all users "
        "with the SQL query tool, then search for documentation about user "
        "management with the search tool. Use both tools."
    )

    print(f"\nPrompt: {prompt}\n")
    print(f"Tools available to LLM: {list(tools.keys())}\n")

    # Note: Tools must be explicitly passed via ModelOption.TOOLS for the LLM to use them.
    # Tool extraction and LLM tool availability are separate concerns.
    response = session.instruct(
        prompt,
        model_options={
            ModelOption.TOOLS: tools,
            ModelOption.TOOL_CHOICE: "auto",
            ModelOption.MAX_NEW_TOKENS: 1000,
        },
        strategy=None,
        tool_calls=True,
    )

    print(f"LLM Response:\n{response.value}\n")

    # Check if tool calls were requested
    if response.tool_calls:
        print(f"Tool calls requested by LLM: {list(response.tool_calls.keys())}")
        print("\nExecuting tool calls via Mellea's pipeline (hooks enabled):")
        # Execute tools through Mellea's pipeline to trigger telemetry hooks
        tool_messages = _call_tools(response, backend)
        for msg in tool_messages:
            print(f"  {msg.name}() → {msg.content}")
        print("\nNote: Tool calls are executed via _call_tools() so telemetry")
        print("hooks fire and metrics are recorded in mellea.tool.calls")
    else:
        print("No tool calls were requested by the LLM.")

    print("\n" + "=" * 70)
    print("PART 3: Tool Attributes Demonstration")
    print("=" * 70)

    # Demonstrate that component IDs are embedded in tool names and component
    # metadata is available from the tool extraction
    print("\nTool name analysis (component ID is embedded in the prefix):")
    for tool_name in sorted(query_tools):
        # Extract component ID from the prefixed tool name
        # Format: component_{component_id}.{tool_name}
        if tool_name.startswith("component_"):
            parts = tool_name.split(".", 1)
            if len(parts) == 2:
                component_id = parts[0].replace("component_", "")
                original_name = parts[1]
                print(f"  {tool_name}")
                print(f"    → component_id: {component_id}")
                print(f"    → original_name: {original_name}")

    print("\n" + "=" * 70)
    print("✓ Successfully demonstrated component ID-based tool prefixing")
    print("✓ Tool calling telemetry enabled (see MELLEA_METRICS* env vars above)")
    print("=" * 70)


if __name__ == "__main__":
    main()
