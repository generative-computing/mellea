"""Hook payload policies for Mellea hooks."""

from __future__ import annotations

try:
    from mcpgateway.plugins.framework.hooks.policies import HookPayloadPolicy

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False


def _build_policies() -> dict[str, object]:
    """Build per-hook-type payload modification policies.

    Hooks absent from this table are observe-only.  With ``DefaultHookPolicy.DENY``
    (the Mellea default), any modification attempt on an observe-only hook is rejected.
    """
    if not _HAS_PLUGIN_FRAMEWORK:
        return {}

    return {
        # Session Lifecycle
        "session_pre_init": HookPayloadPolicy(
            writable_fields=frozenset(
                {"backend_name", "model_id", "model_options", "backend_kwargs"}
            )
        ),
        # session_post_init, session_reset, session_cleanup: observe-only
        # Component Lifecycle
        "component_pre_create": HookPayloadPolicy(
            writable_fields=frozenset(
                {
                    "description",
                    "images",
                    "requirements",
                    "icl_examples",
                    "grounding_context",
                    "user_variables",
                    "prefix",
                    "template_id",
                }
            )
        ),
        "component_post_create": HookPayloadPolicy(
            writable_fields=frozenset({"component"})
        ),
        "component_pre_execute": HookPayloadPolicy(
            writable_fields=frozenset(
                {
                    "action",
                    "context",
                    "context_view",
                    "requirements",
                    "model_options",
                    "format",
                    "strategy",
                    "tool_calls_enabled",
                }
            )
        ),
        "component_post_success": HookPayloadPolicy(
            writable_fields=frozenset({"result"})
        ),
        # component_post_error: observe-only
        # Generation Pipeline
        "generation_pre_call": HookPayloadPolicy(
            writable_fields=frozenset({"model_options", "tools", "format"})
        ),
        "generation_post_call": HookPayloadPolicy(
            writable_fields=frozenset({"model_output"})
        ),
        "generation_stream_chunk": HookPayloadPolicy(
            writable_fields=frozenset({"chunk", "accumulated"})
        ),
        # Validation
        "validation_pre_check": HookPayloadPolicy(
            writable_fields=frozenset({"requirements", "model_options"})
        ),
        "validation_post_check": HookPayloadPolicy(
            writable_fields=frozenset({"results", "all_passed"})
        ),
        # Sampling Pipeline
        "sampling_loop_start": HookPayloadPolicy(
            writable_fields=frozenset({"loop_budget"})
        ),
        # sampling_iteration: observe-only
        "sampling_repair": HookPayloadPolicy(
            writable_fields=frozenset({"repair_action", "repair_context"})
        ),
        "sampling_loop_end": HookPayloadPolicy(
            writable_fields=frozenset({"final_result"})
        ),
        # Tool Execution
        "tool_pre_invoke": HookPayloadPolicy(writable_fields=frozenset({"tool_args"})),
        "tool_post_invoke": HookPayloadPolicy(
            writable_fields=frozenset({"tool_output"})
        ),
        # adapter_*, context_*, error_occurred: observe-only
    }


MELLEA_HOOK_PAYLOAD_POLICIES: dict[str, object] = _build_policies()
