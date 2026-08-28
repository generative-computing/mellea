# pytest: ollama, e2e
"""Advanced: Using call_tools() with tool execution hooks.

This example demonstrates using TOOL_PRE_INVOKE and TOOL_POST_INVOKE hooks
with call_tools() to observe and control tool execution. Hooks fire even
when using the low-level call_tools() primitive, providing full plugin support.

Run:
  uv run python docs/examples/primitives/call_tools_with_hooks.py
"""

import logging

from mellea import start_session
from mellea.backends import ModelOption, tool
from mellea.plugins import HookType, PluginMode, block, hook, register
from mellea.stdlib.functional import call_tools

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for '{query}': Found 5 results"


@tool
def read_file(path: str) -> str:
    """Read a file (simulated)."""
    return f"File contents of {path}"


# --- Plugin 1: Tool allowlist ---


ALLOWED_TOOLS = frozenset({"search"})


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.CONCURRENT, priority=5)
async def enforce_allowlist(payload, _):
    """Block tools not in the allowlist."""
    if payload.is_control_flow:
        return  # Framework tools are exempt
    tool_name = payload.model_tool_call.name
    if tool_name not in ALLOWED_TOOLS:
        log.warning(f"BLOCKED tool: {tool_name} (not in allowlist)")
        return block(f"Tool '{tool_name}' is not permitted")
    log.info(f"ALLOWED tool: {tool_name}")


# --- Plugin 2: Tool audit logger ---


@hook(HookType.TOOL_POST_INVOKE, mode=PluginMode.FIRE_AND_FORGET)
async def audit_tools(payload, _):
    """Log every tool execution for audit purposes."""
    status = "OK" if payload.success else "ERROR"
    log.info(
        f"[AUDIT] tool={payload.model_tool_call.name} "
        f"status={status} latency={payload.execution_time_ms}ms"
    )
    if payload.error:
        log.error(f"[AUDIT] Error: {payload.error}")


def example_hooks_with_call_tools():
    """Demonstrate tool hooks firing with call_tools()."""
    log.info("=" * 60)
    log.info("Example: Hooks with call_tools()")
    log.info("=" * 60)

    with start_session() as m:
        log.info("\n--- Allowed tool (search) ---")
        register(enforce_allowlist)
        register(audit_tools)

        result = m.instruct(
            description="Search for information about Python.",
            model_options={ModelOption.TOOLS: [search, read_file]},
            tool_calls=True,
        )
        log.info(f"Model output: {result.value[:80]}...")

        # call_tools() fires both hooks
        tool_messages = call_tools(result, m.backend)
        log.info(f"Tool execution completed: {len(tool_messages)} tools executed")

        for msg in tool_messages:
            log.info(f"  Tool: {msg.name}, Output: {msg._tool_output}")


def example_plugin_modifications():
    """Show how plugins can modify tool arguments via hooks."""
    import dataclasses

    from mellea.plugins import PluginResult

    log.info("\n" + "=" * 60)
    log.info("Example: Plugin modifies tool arguments")
    log.info("=" * 60)

    @hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.CONCURRENT, priority=10)
    async def sanitize_paths(payload, _):
        """Normalize file paths before execution."""
        if payload.model_tool_call.name != "read_file":
            return
        args = dict(payload.model_tool_call.args or {})
        raw_path = str(args.get("path", ""))

        # Simulate path sanitization
        sanitized_path = raw_path.strip()
        if sanitized_path != raw_path:
            log.info(f"Sanitized path: '{raw_path}' → '{sanitized_path}'")
            new_args = {**args, "path": sanitized_path}
            new_call = dataclasses.replace(payload.model_tool_call, args=new_args)
            return PluginResult(
                continue_processing=True,
                modified_payload=payload.model_copy(
                    update={"model_tool_call": new_call}
                ),
            )

    register(sanitize_paths)

    with start_session() as m:
        result = m.instruct(
            description="Search for information about 'Sanitization example'",
            model_options={ModelOption.TOOLS: [search, read_file]},
            tool_calls=True,
        )
        log.info(f"Model output: {result.value[:80]}...")

        tool_messages = call_tools(result, m.backend)
        log.info(f"Tool execution completed: {len(tool_messages)} tools executed")

        for msg in tool_messages:
            log.info(f"  Tool: {msg.name}, Output: {msg._tool_output}")


if __name__ == "__main__":
    example_hooks_with_call_tools()
    example_plugin_modifications()
    log.info("\n" + "=" * 60)
    log.info("Examples complete")
    log.info("=" * 60)
