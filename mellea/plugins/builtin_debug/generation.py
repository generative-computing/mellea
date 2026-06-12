"""Built-in debug plugin for generation pipeline (pre-call and post-call).

Provides tracing for all LLM backend calls. Use for debugging model invocations,
tracking latency, and understanding request/response flow.

Examples:
    Enable generation tracing:

        from mellea.plugins.builtin_debug.generation import (
            log_generation_pre_call,
            log_generation_post_call,
        )
        from mellea import start_session
        from mellea.plugins import register

        register([log_generation_pre_call, log_generation_post_call])

        with start_session() as m:
            result = m.instruct("...")  # Tracing fires automatically
"""

from __future__ import annotations

import logging
from typing import Any

from mellea.plugins import HookType, hook
from mellea.plugins.hooks.generation import (
    GenerationPostCallPayload,
    GenerationPreCallPayload,
)

logger = logging.getLogger(__name__)


def _get_prompt_preview(payload) -> str:
    """Extract and shorten prompt for logging."""
    action = payload.action
    if not action:
        return "(no action)"

    text = None

    # Try format_for_llm() for structured output
    if hasattr(action, "format_for_llm"):
        try:
            formatted = action.format_for_llm()
            # Extract args if available
            if hasattr(formatted, "args"):
                desc = formatted.args.get("description", "")
                if desc:
                    text = str(desc)
        except (AttributeError, TypeError):
            pass

    # Fallback to string representation
    if not text:
        text = str(action)

    text = text.replace("\n", " ").replace("  ", " ").strip()
    if len(text) > 100:
        text = text[:97] + "..."
    return text


def _get_response_preview(payload) -> str:
    """Extract and shorten response for logging."""
    model_output = payload.model_output
    if not model_output:
        return "(no output)"

    value = model_output.value
    if not value:
        return "(no value)"

    text = str(value).replace("\n", " ").replace("  ", " ").strip()
    if len(text) > 100:
        text = text[:97] + "..."
    return text


def _get_token_usage(payload) -> str:
    """Extract token usage from payload."""
    model_output = payload.model_output
    if not model_output:
        return "unknown"

    gen = model_output.generation
    if not gen:
        return "unknown"

    usage = gen.usage
    if not usage:
        return "unknown"

    total = usage.get("total_tokens", "?")
    prompt = usage.get("prompt_tokens", "?")
    completion = usage.get("completion_tokens", "?")
    return f"({prompt}+{completion}={total})"


@hook(HookType.GENERATION_PRE_CALL)
async def log_generation_pre_call(payload: GenerationPreCallPayload, ctx: Any) -> None:
    """Log request details before calling the LLM.

    Args:
        payload: GenerationPreCallPayload containing action, generation_id.
        ctx: Plugin context for hook execution; backend_name is in global_context.state.
    """
    model_id = "unknown"
    if ctx and hasattr(ctx, "global_context") and ctx.global_context:
        gc_state = getattr(ctx.global_context, "state", {})
        if gc_state:
            model_id = gc_state.get("backend_name", "unknown")
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
    logger.debug(
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
        logger.info("   [⭐ REPAIR ATTEMPT] Repair feedback provided")
        # Show first 100 chars of repair text at DEBUG level to avoid logging PII
        repair_preview = repair_text.replace("\n", " ").replace("  ", " ").strip()
        if len(repair_preview) > 100:
            repair_preview = repair_preview[:97] + "..."
        logger.debug(f"   {repair_preview}")


@hook(HookType.GENERATION_POST_CALL)
async def log_generation_post_call(
    payload: GenerationPostCallPayload, ctx: Any
) -> None:
    """Log response details after LLM returns.

    Args:
        payload: GenerationPostCallPayload containing model_output, generation_id, latency_ms.
        ctx: Plugin context for hook execution; backend_name is in global_context.state.
    """
    model_id = "unknown"
    if ctx and hasattr(ctx, "global_context") and ctx.global_context:
        gc_state = getattr(ctx.global_context, "state", {})
        if gc_state:
            model_id = gc_state.get("backend_name", "unknown")

    gen_id = payload.generation_id or "no-id"
    latency_ms = payload.latency_ms or 0

    response_preview = _get_response_preview(payload)
    tokens = _get_token_usage(payload)

    logger.debug(
        f"[📥 GEN-POST-CALL gen_id={gen_id}] "
        f"model={model_id} | latency={latency_ms:.0f}ms | "
        f"tokens={tokens} | response={response_preview}"
    )
