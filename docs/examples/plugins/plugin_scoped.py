# pytest: ollama, llm
#
# Scoped plugins — activate plugins for a specific block of code.
#
# Three forms of scoped activation are shown:
#
#   1. plugin_scope(*items)  — the universal scope; accepts standalone hooks,
#                              @plugin instances, PluginSets, or any mix
#   2. with <@plugin instance>  — class-based plugin used directly as a context
#                                 manager (equivalent to plugin_scope for one item)
#   3. with <PluginSet>         — a named group used directly as a context manager
#
# All three guarantee that the plugins are registered on __enter__ and
# deregistered on __exit__, even if the block raises an exception.
#
# Run:
#   uv run python docs/examples/plugins/plugin_scoped.py

import logging
import sys

from mellea import start_session
from mellea.plugins import (
    HookType,
    PluginMode,
    PluginSet,
    PluginViolationError,
    hook,
    plugin,
    plugin_scope,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("plugin_scoped")


# ---------------------------------------------------------------------------
# Reusable hook and plugin definitions
# ---------------------------------------------------------------------------


@hook(HookType.COMPONENT_PRE_CREATE, mode=PluginMode.PERMISSIVE, priority=10)
async def log_request(payload, ctx):
    """Log every instruction description as it arrives."""
    preview = payload.description[:60].replace("\n", " ")
    log.info("[log_request] → %r", preview)


@hook(HookType.COMPONENT_POST_SUCCESS, mode=PluginMode.PERMISSIVE, priority=10)
async def log_response(payload, ctx):
    """Log latency after each successful generation."""
    log.info(
        "[log_response] ← %s completed in %dms",
        payload.component_type,
        payload.latency_ms,
    )


@plugin("content-guard", priority=5)
class ContentGuard:
    """Blocks instructions that mention restricted topics."""

    BLOCKED = ["financial advice", "medical diagnosis"]

    @hook(HookType.COMPONENT_PRE_CREATE, mode=PluginMode.ENFORCE, priority=5)
    async def check_description(self, payload, ctx):
        desc = payload.description.lower()
        for topic in self.BLOCKED:
            if topic in desc:
                log.info("[content-guard] BLOCKED: %r", topic)
                from mellea.plugins import block

                return block(f"Restricted topic: {topic}", code="CONTENT_001")
        log.info("[content-guard] description is clean")


observability = PluginSet("observability", [log_request, log_response])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(session, prompt: str) -> str:
    try:
        result = session.instruct(prompt)
        log.info("Result: %s\n", result)
        return str(result)
    except PluginViolationError as e:
        log.error(
            "Blocked on %s: [%s] %s (plugin=%s)\n",
            e.hook_type,
            e.code,
            e.reason,
            e.plugin_name,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with start_session() as m:
        # -------------------------------------------------------------------
        # 1. plugin_scope — the universal scoped activation
        #    Accepts standalone @hook functions, @plugin instances, PluginSets,
        #    or any mix.  Nothing fires outside the block.
        # -------------------------------------------------------------------
        log.info("=== 1. plugin_scope ===")

        log.info("Before scope: no plugins active")
        # (no hooks fire here)

        with plugin_scope(log_request, log_response):
            log.info("Inside plugin_scope: log_request + log_response active")
            run(m, "Name the planets of the solar system.")

        log.info("After plugin_scope: hooks deregistered")
        log.info("")

        # -------------------------------------------------------------------
        # 2. @plugin instance as context manager
        #    A single @plugin-decorated class instance can be entered directly.
        # -------------------------------------------------------------------
        log.info("=== 2. @plugin instance as context manager ===")

        guard = ContentGuard()
        with guard:
            log.info("Inside with guard: ContentGuard active")
            run(m, "What is the boiling point of water?")

        log.info("After with guard: ContentGuard deregistered")
        log.info("")

        # -------------------------------------------------------------------
        # 3. PluginSet as context manager
        #    A PluginSet groups related hooks and can be entered as a unit.
        # -------------------------------------------------------------------
        log.info("=== 3. PluginSet as context manager ===")

        with observability:
            log.info("Inside with observability: log_request + log_response active")
            run(m, "What is the capital of France?")

        log.info("After with observability: hooks deregistered")
        log.info("")

        # -------------------------------------------------------------------
        # 4. Nesting and mixing forms
        #    Scopes stack cleanly — each exit deregisters only its own plugins.
        # -------------------------------------------------------------------
        log.info("=== 4. Nested / mixed scopes ===")

        guard2 = ContentGuard()
        with plugin_scope(log_request):
            log.info("Outer scope: log_request active")
            with guard2:
                log.info("Inner scope: log_request + ContentGuard active")
                run(m, "Briefly explain what a compiler does.")
            log.info("ContentGuard exited — only log_request remains")
            run(m, "What is 12 times 12?")

        log.info("All scopes exited: no plugins active")
        log.info("")

        # -------------------------------------------------------------------
        # 5. Cleanup on exception
        #    Plugins are always deregistered even if the block raises.
        # -------------------------------------------------------------------
        log.info("=== 5. Cleanup on exception ===")

        guard3 = ContentGuard()
        try:
            with guard3:
                log.info("ContentGuard active — raising intentionally")
                raise RuntimeError("deliberate error")
        except RuntimeError:
            pass

        log.info("ContentGuard deregistered despite exception")
        run(m, "What color is the sky?")  # guard3 does not fire here
