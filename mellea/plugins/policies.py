"""Hook payload policies for Mellea hooks."""

from __future__ import annotations

from typing import Any

try:
    from cpex.framework.hooks.policies import (
        HookPayloadPolicy,  # type: ignore[import-not-found]
    )

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False


def _build_policies() -> dict[str, Any]:
    """Build per-hook-type payload modification policies.

    Mutability is enforced at **two layers**:

    1. **Execution mode** (cpex) — only ``SEQUENTIAL`` and ``TRANSFORM`` plugins
       can modify payloads.  ``AUDIT``, ``CONCURRENT``, and ``FIRE_AND_FORGET``
       plugins have their modifications silently discarded by cpex regardless of
       what this table says.

    2. **Field-level policy** (this table) — for modes that *can* modify, this
       table restricts *which* fields are writable.  cpex applies
       ``HookPayloadPolicy`` after each plugin returns, accepting only changes
       to listed fields and discarding the rest.

    Hooks absent from this table are observe-only; with
    ``DefaultHookPolicy.DENY`` (the Mellea default), any modification attempt
    on an unlisted hook is rejected by cpex at runtime.
    """
    if not _HAS_PLUGIN_FRAMEWORK:
        return {}

    return {
        # Session Lifecycle
        "session_pre_init": HookPayloadPolicy(
            writable_fields=frozenset({"model_id", "model_options"})
        ),
        # session_post_init, session_reset, session_cleanup: observe-only
        # Component Lifecycle
        "component_pre_execute": HookPayloadPolicy(
            writable_fields=frozenset(
                {
                    "requirements",
                    "model_options",
                    "format",
                    "strategy",
                    "tool_calls_enabled",
                }
            )
        ),
        # component_post_success, component_post_error: observe-only
        # Generation Pipeline
        "generation_pre_call": HookPayloadPolicy(
            writable_fields=frozenset({"model_options", "tool_calls", "format"})
        ),
        # generation_post_call: observe-only
        # Validation
        "validation_pre_check": HookPayloadPolicy(
            writable_fields=frozenset({"requirements", "model_options"})
        ),
        "validation_post_check": HookPayloadPolicy(
            writable_fields=frozenset({"results", "all_validations_passed"})
        ),
        # Sampling Pipeline
        "sampling_loop_start": HookPayloadPolicy(
            writable_fields=frozenset({"loop_budget"})
        ),
        # sampling_iteration, sampling_repair, sampling_loop_end: observe-only
        # Tool Execution
        "tool_pre_invoke": HookPayloadPolicy(
            writable_fields=frozenset({"model_tool_call"})
        ),
        "tool_post_invoke": HookPayloadPolicy(
            writable_fields=frozenset({"tool_output"})
        ),
        # adapter_*, context_*, error_occurred: observe-only
    }


MELLEA_HOOK_PAYLOAD_POLICIES: dict[str, Any] = _build_policies()
