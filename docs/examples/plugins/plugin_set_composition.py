# pytest: ollama, llm
#
# PluginSet composition — group hooks by concern and register them together.
#
# This example shows how to:
#   1. Define hooks across different concerns (security, observability)
#   2. Group them into PluginSets
#   3. Register observability globally and security per-session
#
# Run:
#   uv run python docs/examples/plugins/plugin_set_composition.py

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("mcpgateway.config").setLevel(logging.ERROR)
logging.getLogger("mcpgateway.observability").setLevel(logging.ERROR)
log = logging.getLogger("plugin_set")

import sys

from mellea import start_session
from mellea.plugins import (
    HookType,
    PluginMode,
    PluginSet,
    PluginViolationError,
    block,
    hook,
    register,
)

# --- Security hooks ---

TOKEN_BUDGET = 4000


@hook(HookType.GENERATION_PRE_CALL, mode=PluginMode.ENFORCE, priority=10)
async def enforce_token_budget(payload, ctx):
    """Enforce a conservative token budget."""
    # Rough token estimate: ~4 chars per token
    prompt_chars = sum(
        len(str(c.format_for_llm()))
        for c in (payload.context.view_for_generation() or [])
    ) + len(
        str(
            payload.action.format_for_llm()
            if hasattr(payload.action, "format_for_llm")
            else payload.action
        )
    )
    estimated = prompt_chars // 4 or 0
    log.info("[security/token-budget] estimated=%d budget=%d", estimated, TOKEN_BUDGET)
    if estimated > TOKEN_BUDGET:
        return block(
            f"Estimated {estimated} tokens exceeds budget of {TOKEN_BUDGET}",
            code="TOKEN_BUDGET_001",
        )


@hook(HookType.COMPONENT_PRE_CREATE, mode=PluginMode.ENFORCE, priority=20)
async def enforce_description_length(payload, ctx):
    """Reject component descriptions that are too long."""
    max_len = 2000
    if len(payload.description) > max_len:
        log.info(
            "[security/desc-length] BLOCKED: description is %d chars",
            len(payload.description),
        )
        return block(
            f"Description exceeds {max_len} characters", code="DESC_LENGTH_001"
        )
    log.info(
        "[security/desc-length] description length OK (%d chars)",
        len(payload.description),
    )


# --- Observability hooks ---


@hook(HookType.SESSION_POST_INIT, mode=PluginMode.PERMISSIVE)
async def trace_session_start(payload, ctx):
    """Trace session initialization."""
    log.info(
        "[observability/trace] session started (session_id=%s)", payload.session_id
    )


@hook(HookType.COMPONENT_POST_SUCCESS, mode=PluginMode.PERMISSIVE)
async def trace_component_success(payload, ctx):
    """Trace successful component executions."""
    log.info(
        "[observability/trace] %s completed in %dms",
        payload.component_type,
        payload.latency_ms,
    )


@hook(HookType.SESSION_CLEANUP, mode=PluginMode.PERMISSIVE)
async def trace_session_end(payload, ctx):
    """Trace session cleanup."""
    log.info(
        "[observability/trace] session cleanup (interactions=%d)",
        payload.interaction_count,
    )


# --- Compose into PluginSets ---

security = PluginSet("security", [enforce_token_budget, enforce_description_length])
observability = PluginSet(
    "observability", [trace_session_start, trace_component_success, trace_session_end]
)


if __name__ == "__main__":
    log.info("--- PluginSet composition example ---")
    log.info("")

    # Register observability globally — fires for all sessions
    register(observability)
    log.info("Registered observability plugins globally")
    log.info("")

    # Session with security plugins (session-scoped) + global observability
    log.info("=== Session with security + observability ===")
    with start_session(plugins=[security]) as m:
        try:
            result = m.instruct("Name three prime numbers.")
            log.info("Result: %s", result)
        except PluginViolationError as e:
            log.error(
                "Execution blocked on %s: [%s] %s (plugin=%s)",
                e.hook_type,
                e.code,
                e.reason,
                e.plugin_name,
            )
            sys.exit(1)
    log.info("")

    log.info("=== Session with observability only ===")
    with start_session() as m:
        try:
            result = m.instruct("What is 2 + 2?")
            log.info("Result: %s", result)
        except PluginViolationError as e:
            log.error(
                "Execution blocked on %s: [%s] %s (plugin=%s)",
                e.hook_type,
                e.code,
                e.reason,
                e.plugin_name,
            )
            sys.exit(1)
