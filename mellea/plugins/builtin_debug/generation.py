"""Built-in debug plugin for generation pipeline (pre-call and post-call).

Provides tracing for all LLM backend calls. Use for debugging model invocations,
tracking latency, and understanding request/response flow.

Examples:
    Enable generation tracing:

        from mellea.plugins.builtin_debug import GenerationTracingPlugin
        from mellea import start_session
        from mellea.plugins import register

        register([GenerationTracingPlugin()])

        with start_session() as m:
            result = m.instruct("...")  # Tracing fires automatically
"""

from __future__ import annotations

import logging

from mellea.plugins import HookType, hook

logger = logging.getLogger(__name__)


def _get_prompt_preview(payload) -> str:
    """Extract and shorten prompt for logging."""
    action = getattr(payload, "action", None)
    if not action:
        return "(no action)"

    text = None

    # Try to get description from Instruction components
    if hasattr(action, "_description"):
        text = action._description or None

    # Try format_for_llm() for structured output
    if not text and hasattr(action, "format_for_llm"):
        try:
            formatted = action.format_for_llm()
            # Extract args if available
            if hasattr(formatted, "args"):
                desc = formatted.args.get("description", "")
                if desc:
                    text = str(desc)[:200]
        except Exception:
            pass

    # Fallback to string representation
    if not text:
        text = str(action)[:200]

    text = str(text).replace("\n", " ").replace("  ", " ").strip()
    if len(text) > 100:
        text = text[:97] + "..."
    return text


def _get_response_preview(payload) -> str:
    """Extract and shorten response for logging."""
    try:
        model_output = getattr(payload, "model_output", None)
        if not model_output:
            return "(no output)"

        value = getattr(model_output, "value", None)
        if not value:
            return "(no value)"

        text = str(value)[:200]
        text = text.replace("\n", " ").replace("  ", " ").strip()
        if len(text) > 100:
            text = text[:97] + "..."
        return text
    except Exception:
        return "(error reading response)"


def _get_token_usage(payload) -> str:
    """Extract token usage from payload."""
    try:
        model_output = getattr(payload, "model_output", None)
        if not model_output:
            return "unknown"

        gen = getattr(model_output, "generation", None)
        if not gen:
            return "unknown"

        usage = getattr(gen, "usage", {})
        if not usage:
            return "unknown"

        total = usage.get("total_tokens", "?")
        prompt = usage.get("prompt_tokens", "?")
        completion = usage.get("completion_tokens", "?")
        return f"({prompt}+{completion}={total})"
    except Exception:
        return "unknown"


@hook(HookType.GENERATION_PRE_CALL)
async def log_generation_pre_call(payload, ctx):
    """Log request details before calling the LLM.

    Args:
        payload: GenerationPreCallPayload containing backend, action, generation_id.
        ctx: Plugin context for hook execution.
    """
    model = getattr(payload, "backend", None)
    model_id = model.model_id if model else "unknown"
    gen_id = payload.generation_id or "no-id"

    # Extract all data from the action
    action = getattr(payload, "action", None)
    requirements = []
    repair_text = ""

    if action and hasattr(action, "format_for_llm"):
        try:
            fmt = action.format_for_llm()
            if hasattr(fmt, "args"):
                requirements = fmt.args.get("requirements", [])
                repair_text = fmt.args.get("repair", "")
        except Exception:
            pass

    # Log main request info
    prompt_preview = _get_prompt_preview(payload)
    logger.info(
        f"[📤 GEN-PRE-CALL gen_id={gen_id}] model={model_id} | prompt={prompt_preview}"
    )

    # Log requirements if present
    if requirements:
        logger.debug(f"   requirements ({len(requirements)}):")
        for i, req in enumerate(requirements, 1):
            req_desc = getattr(req, "description", str(req))
            logger.debug(f"     {i}. {req_desc}")

    # Log repair feedback if present (indicates a repair attempt)
    if repair_text:
        logger.info("   [⭐ REPAIR ATTEMPT] Repair feedback provided:")
        # Show first 300 chars of repair text
        repair_preview = repair_text[:300].replace("\n", " ")
        if len(repair_text) > 300:
            repair_preview += "..."
        logger.info(f"   {repair_preview}")


@hook(HookType.GENERATION_POST_CALL)
async def log_generation_post_call(payload, ctx):
    """Log response details after LLM returns.

    Args:
        payload: GenerationPostCallPayload containing model_output, generation_id, latency_ms.
        ctx: Plugin context for hook execution.
    """
    model_output = getattr(payload, "model_output", None)
    model_id = "unknown"
    if model_output:
        gen = getattr(model_output, "generation", None)
        if gen:
            model_id = getattr(gen, "model", "unknown")

    gen_id = payload.generation_id or "no-id"
    latency_ms = payload.latency_ms or 0

    response_preview = _get_response_preview(payload)
    tokens = _get_token_usage(payload)

    logger.info(
        f"[📥 GEN-POST-CALL gen_id={gen_id}] "
        f"model={model_id} | latency={latency_ms:.0f}ms | "
        f"tokens={tokens} | response={response_preview}"
    )


# Export as a name for convenience
class GenerationTracingPlugin:
    """Marker class for GenerationTracingPlugin.

    The actual hooks are registered via @hook decorators above.
    Use this for reference or type hinting if needed.
    """
