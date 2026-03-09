# pytest: ollama, llm
#
# Execution modes — all four PluginMode values side by side.
#
# This example registers four hooks on the same hook type
# (COMPONENT_PRE_EXECUTE), each using a different execution mode.
# It demonstrates:
#
#   1. SEQUENTIAL   — runs inline, can block or modify
#   2. CONCURRENT   — runs inline alongside other concurrent hooks
#   3. AUDIT        — runs inline, violations logged but not enforced
#   4. FIRE_AND_FORGET — runs in background, result ignored
#
# Run:
#   uv run python docs/examples/plugins/execution_modes.py

import logging

from mellea import start_session
from mellea.plugins import (
    HookType,
    PluginMode,
    PluginViolationError,
    block,
    hook,
    plugin_scope,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("fancy_logger").setLevel(logging.ERROR)
log = logging.getLogger("execution_modes")


# --- Hook 1: SEQUENTIAL (priority=10) ---
# Runs first, inline. Could block execution if it returned block().

@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.SEQUENTIAL, priority=10)
async def sequential_hook(payload, ctx):
    """Sequential hook — runs inline in priority order."""
    log.info("[SEQUENTIAL  p=10] component=%s", payload.component_type)


# --- Hook 2: CONCURRENT (priority=20) ---
# Dispatched concurrently with other concurrent hooks at the same priority.

@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.CONCURRENT, priority=20)
async def concurrent_hook(payload, ctx):
    """Concurrent hook — runs inline but concurrently with peers."""
    log.info("[CONCURRENT  p=20] component=%s", payload.component_type)


# --- Hook 3: AUDIT (priority=30) ---
# Runs inline; violations are logged but do NOT block execution.

@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.AUDIT, priority=30)
async def audit_hook(payload, ctx):
    """Audit hook — violation is logged but does not block."""
    log.info("[AUDIT       p=30] would block, but audit mode only logs")
    return block("Audit-mode violation: for monitoring only", code="AUDIT_001")


# --- Hook 4: FIRE_AND_FORGET (priority=40) ---
# Dispatched via asyncio.create_task(); result is ignored.
# The log line may appear *after* the main result is printed.

@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.FIRE_AND_FORGET, priority=40)
async def fire_and_forget_hook(payload, ctx):
    """Fire-and-forget hook — runs in background, never blocks."""
    log.info("[FIRE_FORGET p=40] logging in the background")


if __name__ == "__main__":
    log.info("--- Execution modes example ---")
    log.info("")

    with start_session() as m:
        with plugin_scope(
            sequential_hook, concurrent_hook, audit_hook, fire_and_forget_hook
        ):
            try:
                result = m.instruct("Name the four seasons.")
                log.info("")
                log.info("Result: %s", result)
            except PluginViolationError as e:
                log.error("Blocked: %s", e)

    log.info("")
    log.info(
        "Note: the FIRE_AND_FORGET log may have appeared after the result "
        "— that is expected behavior."
    )
