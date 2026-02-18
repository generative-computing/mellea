# pytest: ollama, llm
#
# Standalone function hooks — the simplest way to extend Mellea.
#
# This example registers two function-based hooks:
#   1. A generation_pre_call hook that logs and enforces a token budget
#   2. A component_post_success hook that logs generation latency
#
# Run:
#   uv run python docs/examples/plugins/standalone_hooks.py

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("mcpgateway.config").setLevel(logging.ERROR)
logging.getLogger("mcpgateway.observability").setLevel(logging.ERROR)
log = logging.getLogger("standalone_hooks")

import sys

from mellea import start_session
from mellea.plugins import MelleaHookType, PluginViolationError, block, hook, register

TOKEN_BUDGET = 4000


@hook(MelleaHookType.GENERATION_PRE_CALL, mode="enforce", priority=10)
async def enforce_token_budget(payload, ctx):
    """Block generation calls that exceed the token budget."""
    estimated = payload.estimated_tokens or 0
    log.info(
        "[enforce_token_budget] estimated_tokens=%d, budget=%d", estimated, TOKEN_BUDGET
    )
    if estimated > TOKEN_BUDGET:
        return block(
            f"Estimated {estimated} tokens exceeds budget of {TOKEN_BUDGET}",
            code="TOKEN_BUDGET_001",
            details={"estimated": estimated, "budget": TOKEN_BUDGET},
        )
    log.info("[enforce_token_budget] within budget — allowing generation")


@hook(MelleaHookType.COMPONENT_POST_SUCCESS, mode="permissive", priority=50)
async def log_latency(payload, ctx):
    """Log latency after each successful component execution."""
    log.info(
        "[log_latency] component=%s latency=%dms",
        payload.component_type,
        payload.latency_ms,
    )


# Register both hooks globally — they fire for every session
register([enforce_token_budget, log_latency])

if __name__ == "__main__":
    log.info("--- Standalone function hooks example ---")

    with start_session() as m:
        log.info("Session started (id=%s)", m.id)
        log.info("")

        try:
            result = m.instruct("What are the three laws of robotics? Be brief.")
            log.info("")
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
