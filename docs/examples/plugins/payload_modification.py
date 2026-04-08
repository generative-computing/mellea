# pytest: ollama, e2e
#
# Payload modification — how to modify payloads in hooks.
#
# This example demonstrates:
#   1. Using modify() to change writable payload fields
#   2. Using model_copy(update={...}) directly for fine-grained control
#   3. What happens when you try to modify a non-writable field (silently discarded)
#
# Run:
#   uv run python docs/examples/plugins/payload_modification.py

import logging

from mellea import start_session
from mellea.plugins import (
    HookType,
    PluginMode,
    PluginResult,
    hook,
    modify,
    plugin_scope,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("fancy_logger").setLevel(logging.ERROR)
log = logging.getLogger("payload_modification")


# ---------------------------------------------------------------------------
# Hook 1: Inject a max_tokens cap via modify() helper
#
# generation_pre_call writable fields include: model_options, format, tool_calls
# ---------------------------------------------------------------------------


@hook(HookType.GENERATION_PRE_CALL, mode=PluginMode.SEQUENTIAL, priority=10)
async def cap_max_tokens(payload, ctx):
    """Cap max_tokens to 256 on every generation call."""
    opts = dict(payload.model_options or {})
    if opts.get("max_tokens", float("inf")) > 256:
        log.info("[cap_max_tokens] capping max_tokens to 256")
        opts["max_tokens"] = 256
        return modify(payload, model_options=opts)
    log.info("[cap_max_tokens] max_tokens already within cap")


# ---------------------------------------------------------------------------
# Hook 2: Inject default model options via modify()
#
# component_pre_execute writable fields include: requirements, model_options, ...
# This shows model_copy(update={...}) for fine-grained control.
# ---------------------------------------------------------------------------


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.SEQUENTIAL, priority=10)
async def inject_default_options(payload, ctx):
    """Ensure a default temperature is set on every component execution."""
    opts = dict(payload.model_options or {})
    if "temperature" not in opts:
        log.info("[inject_default_options] setting default temperature=0.7")
        opts["temperature"] = 0.7
        return modify(payload, model_options=opts)
    log.info("[inject_default_options] temperature already set")


# ---------------------------------------------------------------------------
# Hook 3: Attempt to modify a non-writable field (observe it is discarded)
#
# generation_pre_call does NOT include 'action' or 'context' as writable.
# This hook tries to modify 'context' — the change will be silently discarded
# by the payload policy enforcement, and the original context will be used.
# ---------------------------------------------------------------------------


@hook(HookType.GENERATION_PRE_CALL, mode=PluginMode.SEQUENTIAL, priority=20)
async def attempt_non_writable(payload, ctx):
    """Try to modify a non-writable field — change will be silently discarded."""
    log.info("[attempt_non_writable] attempting to modify 'hook' (non-writable)")
    # This modification will be filtered out by the payload policy
    modified = payload.model_copy(update={"hook": "tampered"})
    return PluginResult(continue_processing=True, modified_payload=modified)


if __name__ == "__main__":
    log.info("--- Payload modification example ---")
    log.info("")

    with start_session() as m:
        with plugin_scope(cap_max_tokens, inject_default_options, attempt_non_writable):
            result = m.instruct(
                "Summarize the benefits of open-source software in one sentence."
            )
            log.info("")
            log.info("Result: %s", result)

    log.info("")
    log.info(
        "Note: the 'hook' field modification in attempt_non_writable was silently "
        "discarded by the payload policy — only writable fields are accepted."
    )
