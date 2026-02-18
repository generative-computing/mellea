"""Mellea hook type enum and hook registration."""

from __future__ import annotations

from enum import Enum
from typing import Any

try:
    from mcpgateway.plugins.framework.hooks.registry import get_hook_registry
    from mcpgateway.plugins.framework.models import PluginResult

    _HAS_PLUGIN_FRAMEWORK = True
except ImportError:
    _HAS_PLUGIN_FRAMEWORK = False


class PluginMode(str, Enum):
    """Execution modes for Mellea plugins."""

    ENFORCE = "enforce"
    PERMISSIVE = "permissive"
    FIRE_AND_FORGET = "fire_and_forget"


class HookType(str, Enum):
    """All Mellea hook types."""

    # Session Lifecycle
    SESSION_PRE_INIT = "session_pre_init"
    SESSION_POST_INIT = "session_post_init"
    SESSION_RESET = "session_reset"
    SESSION_CLEANUP = "session_cleanup"

    # Component Lifecycle
    COMPONENT_PRE_CREATE = "component_pre_create"
    COMPONENT_POST_CREATE = "component_post_create"
    COMPONENT_PRE_EXECUTE = "component_pre_execute"
    COMPONENT_POST_SUCCESS = "component_post_success"
    COMPONENT_POST_ERROR = "component_post_error"

    # Generation Pipeline
    GENERATION_PRE_CALL = "generation_pre_call"
    GENERATION_POST_CALL = "generation_post_call"
    GENERATION_STREAM_CHUNK = "generation_stream_chunk"

    # Validation
    VALIDATION_PRE_CHECK = "validation_pre_check"
    VALIDATION_POST_CHECK = "validation_post_check"

    # Sampling Pipeline
    SAMPLING_LOOP_START = "sampling_loop_start"
    SAMPLING_ITERATION = "sampling_iteration"
    SAMPLING_REPAIR = "sampling_repair"
    SAMPLING_LOOP_END = "sampling_loop_end"

    # Tool Execution
    TOOL_PRE_INVOKE = "tool_pre_invoke"
    TOOL_POST_INVOKE = "tool_post_invoke"

    # Backend Adapter Ops
    ADAPTER_PRE_LOAD = "adapter_pre_load"
    ADAPTER_POST_LOAD = "adapter_post_load"
    ADAPTER_PRE_UNLOAD = "adapter_pre_unload"
    ADAPTER_POST_UNLOAD = "adapter_post_unload"

    # Context Operations
    CONTEXT_UPDATE = "context_update"
    CONTEXT_PRUNE = "context_prune"

    # Error Handling
    ERROR_OCCURRED = "error_occurred"


# Lazily populated mapping: hook_type -> (payload_class, result_class).
# Populated by _build_hook_registry() on first call to _register_mellea_hooks().
_HOOK_REGISTRY: dict[str, tuple[type, type]] = {}


def _build_hook_registry() -> dict[str, tuple[type, type]]:
    """Build the mapping from hook types to (payload_class, PluginResult).

    Imports payload classes lazily to avoid circular imports.
    """
    from mellea.plugins.hooks.component import (
        ComponentPostCreatePayload,
        ComponentPostErrorPayload,
        ComponentPostSuccessPayload,
        ComponentPreCreatePayload,
        ComponentPreExecutePayload,
    )
    from mellea.plugins.hooks.generation import (
        GenerationPostCallPayload,
        GenerationPreCallPayload,
        GenerationStreamChunkPayload,
    )
    from mellea.plugins.hooks.sampling import (
        SamplingIterationPayload,
        SamplingLoopEndPayload,
        SamplingLoopStartPayload,
        SamplingRepairPayload,
    )
    from mellea.plugins.hooks.session import (
        SessionCleanupPayload,
        SessionPostInitPayload,
        SessionPreInitPayload,
        SessionResetPayload,
    )
    from mellea.plugins.hooks.validation import (
        ValidationPostCheckPayload,
        ValidationPreCheckPayload,
    )

    return {
        # Session Lifecycle
        HookType.SESSION_PRE_INIT.value: (SessionPreInitPayload, PluginResult),
        HookType.SESSION_POST_INIT.value: (SessionPostInitPayload, PluginResult),
        HookType.SESSION_RESET.value: (SessionResetPayload, PluginResult),
        HookType.SESSION_CLEANUP.value: (SessionCleanupPayload, PluginResult),
        # Component Lifecycle
        HookType.COMPONENT_PRE_CREATE.value: (ComponentPreCreatePayload, PluginResult),
        HookType.COMPONENT_POST_CREATE.value: (
            ComponentPostCreatePayload,
            PluginResult,
        ),
        HookType.COMPONENT_PRE_EXECUTE.value: (
            ComponentPreExecutePayload,
            PluginResult,
        ),
        HookType.COMPONENT_POST_SUCCESS.value: (
            ComponentPostSuccessPayload,
            PluginResult,
        ),
        HookType.COMPONENT_POST_ERROR.value: (ComponentPostErrorPayload, PluginResult),
        # Generation Pipeline
        HookType.GENERATION_PRE_CALL.value: (GenerationPreCallPayload, PluginResult),
        HookType.GENERATION_POST_CALL.value: (GenerationPostCallPayload, PluginResult),
        HookType.GENERATION_STREAM_CHUNK.value: (
            GenerationStreamChunkPayload,
            PluginResult,
        ),
        # Validation
        HookType.VALIDATION_PRE_CHECK.value: (ValidationPreCheckPayload, PluginResult),
        HookType.VALIDATION_POST_CHECK.value: (
            ValidationPostCheckPayload,
            PluginResult,
        ),
        # Sampling Pipeline
        HookType.SAMPLING_LOOP_START.value: (SamplingLoopStartPayload, PluginResult),
        HookType.SAMPLING_ITERATION.value: (SamplingIterationPayload, PluginResult),
        HookType.SAMPLING_REPAIR.value: (SamplingRepairPayload, PluginResult),
        HookType.SAMPLING_LOOP_END.value: (SamplingLoopEndPayload, PluginResult),
    }


def _register_mellea_hooks() -> None:
    """Register all Mellea hook types with the ContextForge HookRegistry.

    Idempotent — skips already-registered hook types.
    """
    if not _HAS_PLUGIN_FRAMEWORK:
        return

    global _HOOK_REGISTRY
    if not _HOOK_REGISTRY:
        _HOOK_REGISTRY = _build_hook_registry()

    registry: Any = get_hook_registry()
    for hook_type, (payload_cls, result_cls) in _HOOK_REGISTRY.items():
        if not registry.is_registered(hook_type):
            registry.register_hook(hook_type, payload_cls, result_cls)
