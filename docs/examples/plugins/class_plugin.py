# pytest: ollama, e2e
#
# Class-based plugin — group related hooks in a single Plugin subclass.
#
# This example creates a PII protection plugin that:
#   1. Blocks input containing SSN patterns before component execution
#   2. Scans LLM output for SSN patterns after generation (observe-only)
#
# Run:
#   uv run python docs/examples/plugins/class_plugin.py

import logging
import re
import sys

from mellea import start_session
from mellea.plugins import HookType, Plugin, PluginViolationError, block, hook, register

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("fancy_logger").setLevel(logging.ERROR)
log = logging.getLogger("class_plugin")


class PIIRedactor(Plugin, name="pii-redactor", priority=5):
    """Redacts PII patterns from both input and output.

    .. warning:: Shared mutable state
        ``redaction_count`` is shared across all hook invocations.  This is
        safe today because all hooks run on the same ``asyncio`` event loop,
        but would require a lock or ``contextvars`` if hooks ever execute in
        parallel threads.
    """

    def __init__(self, patterns: list[str] | None = None):
        self.patterns = patterns or [
            r"\d{3}-\d{2}-\d{4}",  # SSN
            r"\b\d{16}\b",  # credit card (simplified)
        ]
        self.redaction_count = 0

    @hook(HookType.COMPONENT_PRE_EXECUTE)
    async def reject_pii_input(self, payload, ctx):
        """Block component execution if the action contains PII patterns."""
        if payload.component_type != "Instruction":
            return
        original = (
            str(payload.action._description) if payload.action._description else ""
        )
        if self._contains_pii(original):
            log.warning("[pii-redactor] PII detected in component action — blocking")
            self.redaction_count += 1
            return block(
                "Input contains PII patterns that must be removed before processing",
                code="PII_INPUT_DETECTED",
            )
        log.info("[pii-redactor] no PII found in input")

    @hook(HookType.GENERATION_POST_CALL)
    async def scan_output(self, payload, ctx):
        """Scan LLM output for PII and log a warning if detected.

        ``generation_post_call`` is observe-only — plugins cannot modify the
        ``model_output``.  This hook therefore only inspects the output and
        records a warning for downstream monitoring/alerting.
        """
        mot_value = getattr(payload.model_output, "value", None)
        if mot_value is None:
            log.info("[pii-redactor] output not yet computed — skipping output scan")
            return
        original = str(mot_value)
        if self._contains_pii(original):
            log.warning("[pii-redactor] PII detected in LLM output (observe-only)")
            self.redaction_count += 1
        else:
            log.info("[pii-redactor] no PII found in output")

    def _contains_pii(self, text: str) -> bool:
        return any(re.search(p, text) for p in self.patterns)


# Create an instance and register it globally
redactor = PIIRedactor()
register(redactor)

if __name__ == "__main__":
    log.info("--- Class-based Plugin example (PII Redactor) ---")
    log.info("")

    with start_session() as m:
        log.info("Session started (id=%s)", m.id)
        log.info("")

        # Request 1: contains an SSN — the input hook blocks execution.
        log.info("Request 1: input with PII (should be blocked)")
        try:
            m.instruct(
                "Summarize this customer record: "
                "Name: Jane Doe, SSN: 123-45-6789, Status: Active"
            )
        except PluginViolationError as e:
            log.info(
                "Blocked as expected on %s: [%s] %s", e.hook_type, e.code, e.reason
            )
        log.info("")

        # Request 2: clean input — no PII, so it reaches the LLM.
        # If the LLM output contains PII, scan_output logs a warning (observe-only).
        log.info("Request 2: clean input (should succeed)")
        try:
            result = m.instruct("Name the three primary colors.")
            log.info("Result: %s", result)
        except PluginViolationError as e:
            log.warning(
                "Execution blocked on %s: [%s] %s (plugin=%s)",
                e.hook_type,
                e.code,
                e.reason,
                e.plugin_name,
            )
            sys.exit(1)

        log.info("")
        log.info("Total PII detections: %d", redactor.redaction_count)
