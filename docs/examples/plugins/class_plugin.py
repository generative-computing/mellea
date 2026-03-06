# pytest: ollama, llm
#
# Class-based plugin — group related hooks in a single class with @plugin.
#
# This example creates a PII redaction plugin that:
#   1. Scans input descriptions for SSN patterns before component creation
#   2. Scans LLM output for SSN patterns after generation
#
# Run:
#   uv run python docs/examples/plugins/class_plugin.py

import logging
import re
import sys

from mellea import start_session
from mellea.plugins import (
    HookType,
    PluginResult,
    PluginViolationError,
    hook,
    plugin,
    register,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("mcpgateway").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("fancy_logger").setLevel(logging.ERROR)
log = logging.getLogger("class_plugin")


@plugin("pii-redactor", priority=5)
class PIIRedactor:
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

    @hook(HookType.COMPONENT_PRE_CREATE)
    async def redact_input(self, payload, ctx):
        """Scan and redact PII from component descriptions before they reach the LLM."""
        original = payload.description
        redacted = self._redact(original)
        if redacted != original:
            log.info("[pii-redactor] redacted PII from input description")
            self.redaction_count += 1
            modified = payload.model_copy(update={"description": redacted})
            return PluginResult(continue_processing=True, modified_payload=modified)
        log.info("[pii-redactor] no PII found in input")

    @hook(HookType.GENERATION_POST_CALL)
    async def redact_output(self, payload, ctx):
        """Scan LLM output for PII and log a warning if found.

        Note: This hook fires while the ``ModelOutputThunk`` is still lazy
        (uncomputed), so ``payload.model_output.value`` may be ``None``.
        Modifying the thunk's value from here is not currently supported —
        use ``COMPONENT_PRE_CREATE`` to redact *inputs* before they reach the
        LLM instead.
        """
        mot_value = getattr(payload.model_output, "value", None)
        if mot_value is None:
            log.info("[pii-redactor] output not yet computed — skipping output scan")
            return
        original = str(mot_value)
        redacted = self._redact(original)
        if redacted != original:
            log.warning("[pii-redactor] PII detected in LLM output")
            self.redaction_count += 1
        else:
            log.info("[pii-redactor] no PII found in output")

    def _redact(self, text: str) -> str:
        for pattern in self.patterns:
            text = re.sub(pattern, "[REDACTED]", text)
        return text


# Create an instance and register it globally
redactor = PIIRedactor()
register(redactor)

if __name__ == "__main__":
    log.info("--- Class-based @plugin example (PII Redactor) ---")
    log.info("")

    with start_session() as m:
        log.info("Session started (id=%s)", m.id)
        log.info("")

        try:
            # The SSN in this prompt will be redacted before reaching the LLM
            result = m.instruct(
                "Summarize this customer record: "
                "Name: Jane Doe, SSN: 123-45-6789, Status: Active"
            )
            log.info("")
            log.info("Result: %s", result)
            log.info("")
            log.info("Total redactions applied: %d", redactor.redaction_count)
        except PluginViolationError as e:
            log.warning(
                "Execution blocked on %s: [%s] %s (plugin=%s)",
                e.hook_type,
                e.code,
                e.reason,
                e.plugin_name,
            )
            sys.exit(1)
