# pytest: ollama, e2e
#
# Execution modes — all five PluginMode values side by side.
#
# This example registers five hooks on the same hook type
# (COMPONENT_PRE_EXECUTE), each using a different execution mode.
# It demonstrates:
#
#   1. SEQUENTIAL      — serial, can block + modify
#   2. TRANSFORM       — serial, can modify only (blocks suppressed)
#   3. AUDIT           — serial, observe-only (modifications discarded, blocks logged)
#   4. CONCURRENT      — parallel, can block only (modifications discarded)
#   5. FIRE_AND_FORGET — background, observe-only (result ignored)
#
# Execution order: SEQUENTIAL → TRANSFORM → AUDIT → CONCURRENT → FIRE_AND_FORGET
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
log = logging.getLogger("execution_modes")


# --- Hook 1: SEQUENTIAL (priority=10) ---
# Serial, chained execution. Can block the pipeline and modify writable
# payload fields. Each hook receives the payload from the prior one.


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.SEQUENTIAL, priority=10)
async def sequential_hook(payload, ctx):
    """Sequential hook — can block + modify, runs inline in priority order."""
    log.info("[SEQUENTIAL      p=10] component=%s", payload.component_type)


# --- Hook 2: TRANSFORM (priority=20) ---
# Serial, chained execution after all SEQUENTIAL hooks. Can modify writable
# payload fields but CANNOT block — blocking results are suppressed with a
# warning. Ideal for data transformation (PII redaction, prompt rewriting).


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.TRANSFORM, priority=20)
async def transform_hook(payload, ctx):
    """Transform hook — can modify but cannot block."""
    log.info("[TRANSFORM       p=20] enriching model_options")
    opts = dict(payload.model_options or {})
    opts.setdefault("temperature", 0.7)
    return modify(payload, model_options=opts)


# --- Hook 3: AUDIT (priority=30) ---
# Serial execution after TRANSFORM. Observe-only: payload modifications are
# discarded and violations are logged but do NOT block. Use for monitoring,
# metrics, and gradual policy rollout.


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.AUDIT, priority=30)
async def audit_hook(payload, ctx):
    """Audit hook — observe-only; violations logged but not enforced."""
    log.info("[AUDIT           p=30] would block, but audit mode only logs")
    return block("Audit-mode violation: for monitoring only", code="AUDIT_001")


# --- Hook 4: CONCURRENT (priority=40) ---
# Dispatched in parallel after AUDIT. Can block the pipeline (fail-fast on
# first blocking result) but payload modifications are discarded to avoid
# non-deterministic last-writer-wins races.


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.CONCURRENT, priority=40)
async def concurrent_hook(payload, ctx):
    """Concurrent hook — can block but cannot modify, runs in parallel."""
    log.info("[CONCURRENT      p=40] component=%s", payload.component_type)


# --- Hook 5: FIRE_AND_FORGET (priority=50) ---
# Dispatched via asyncio.create_task() after all other phases. Receives a
# copy-on-write snapshot of the payload. Cannot modify payloads or block
# execution. Any exceptions are logged but do not propagate.
# The log line may appear *after* the main result is printed.


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.FIRE_AND_FORGET, priority=50)
async def fire_and_forget_hook(payload, ctx):
    """Fire-and-forget hook — runs in background, never blocks."""
    log.info("[FIRE_AND_FORGET p=50] logging in the background")


if __name__ == "__main__":
    log.info("--- Execution modes example ---")
    log.info("")

    with start_session() as m:
        with plugin_scope(
            sequential_hook,
            transform_hook,
            audit_hook,
            concurrent_hook,
            fire_and_forget_hook,
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
