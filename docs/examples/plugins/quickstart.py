# pytest: ollama, e2e
#
# Quick Start — your first Mellea plugin in under 30 lines.
#
# This example registers a single function hook that logs every generation
# call, then runs a normal instruct() to show it in action.
#
# Run:
#   uv run python docs/examples/plugins/quickstart.py

import logging

from mellea import start_session
from mellea.plugins import HookType, hook, register

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("fancy_logger").setLevel(logging.ERROR)
log = logging.getLogger("quickstart")


@hook(HookType.GENERATION_PRE_CALL)
async def log_generation(payload, ctx):
    """Log a one-line summary before every LLM call."""
    action_preview = str(payload.action)[:80].replace("\n", " ")
    log.info("[log_generation] About to call LLM: %r", action_preview)


register(log_generation)

if __name__ == "__main__":
    with start_session() as m:
        result = m.instruct("What is the capital of France?")
        log.info("Result: %s", result)
