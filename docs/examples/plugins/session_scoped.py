# pytest: ollama, llm
#
# Session-scoped plugins — plugins that fire only within a specific session.
#
# This example demonstrates:
#   1. A global observability hook that fires for ALL sessions
#   2. A session-scoped content policy that only fires for one session
#   3. A second session without the content policy to show the difference
#
# Run:
#   uv run python docs/examples/plugins/session_scoped.py

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("mcpgateway.config").setLevel(logging.ERROR)
logging.getLogger("mcpgateway.observability").setLevel(logging.ERROR)
log = logging.getLogger("session_scoped")

import sys

from mellea import start_session
from mellea.plugins import (
    HookType,
    PluginMode,
    PluginViolationError,
    block,
    hook,
    register,
)


# --- Global hook: fires for every session ---


@hook(HookType.SESSION_POST_INIT, mode=PluginMode.PERMISSIVE)
async def log_session_init(payload, ctx):
    """Log when any session is initialized."""
    log.info("[global] session initialized (session_id=%s)", payload.session_id)


register(log_session_init)


# --- Session-scoped hooks: passed via start_session(plugins=...) ---

BLOCKED_TOPICS = ["weaponry", "explosives"]


@hook(HookType.COMPONENT_PRE_CREATE, mode=PluginMode.ENFORCE, priority=10)
async def enforce_content_policy(payload, ctx):
    """Block components whose descriptions mention restricted topics."""
    desc = payload.description.lower()
    for topic in BLOCKED_TOPICS:
        if topic in desc:
            log.info("[content-policy] BLOCKED: topic '%s' found in description", topic)
            return block(f"Restricted topic: {topic}", code="CONTENT_POLICY_001")
    log.info("[content-policy] description is clean — allowing")


@hook(HookType.COMPONENT_POST_SUCCESS, mode=PluginMode.PERMISSIVE)
async def log_component_result(payload, ctx):
    """Log the result of each successful component execution."""
    log.info(
        "[session-logger] component=%s latency=%dms",
        payload.component_type,
        payload.latency_ms,
    )


if __name__ == "__main__":
    log.info("--- Session-scoped plugins example ---")
    log.info("")

    # Session 1: has content policy + result logging (session-scoped)
    log.info("=== Session 1 (with content policy) ===")
    with start_session(plugins=[enforce_content_policy, log_component_result]) as m:
        try:
            result = m.instruct("Explain photosynthesis in one sentence.")
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

    # Session 2: only global hooks fire (no content policy)
    log.info("=== Session 2 (global hooks only, no content policy) ===")
    with start_session() as m:
        try:
            result = m.instruct("What is the speed of light?")
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
