# pytest: ollama, e2e
"""Built-in generation tracing plugin example.

This example demonstrates the GenerationTracingPlugin from mellea.plugins.builtin_debug,
which traces all LLM backend calls with request/response inspection.

The plugin logs:
- Generation ID for correlation
- Model being called
- Prompt preview (first 100 chars)
- Response preview (first 100 chars)
- Latency in milliseconds
- Token usage (prompt + completion = total)

Run:
    uv run python docs/examples/plugins/builtin_generation_tracing.py

Watch the logs to see tracing in action:
    [📤 GEN-PRE-CALL gen_id=...] model=... | prompt=...
    [📥 GEN-POST-CALL gen_id=...] model=... | latency=...ms | tokens=(...) | response=...
"""

import logging

import mellea
from mellea.plugins import plugin_scope
from mellea.plugins.builtin_debug.generation import (
    log_generation_post_call,
    log_generation_pre_call,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    """Example: Use generation tracing to debug generation calls.

    The plugin_scope context manager ensures hooks are registered only
    for this block and are cleaned up automatically on exit.
    """
    log.info("=" * 70)
    log.info("Generation Tracing Plugin Example")
    log.info("=" * 70)
    log.info("")
    log.info("Watch the logs for [📤 GEN-PRE-CALL] and [📥 GEN-POST-CALL] entries.")
    log.info("")

    with plugin_scope([log_generation_pre_call, log_generation_post_call]):
        with mellea.start_session() as m:
            result = m.instruct("What are the three main colors of the rainbow?")

            log.info("")
            log.info("=" * 70)
            log.info("Final result:")
            result_str = str(result)
            log.info(result_str[:200] + "..." if len(result_str) > 200 else result_str)
            log.info("=" * 70)


if __name__ == "__main__":
    main()
