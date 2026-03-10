"""Mellea."""

from .backends import model_ids
from .stdlib.components.genslot import generative
from .stdlib.session import MelleaSession, start_session

__all__ = ["MelleaSession", "generative", "model_ids", "start_session"]

# Example PluginSets that attach a hook at every call site.
#
# 1. logging_plugin_set  — observes every hook point and logs it (AUDIT mode)
# 2. blocking_plugin_set — blocks at every hook point (SEQUENTIAL mode)

from mellea.plugins import HookType, PluginMode, PluginSet, block, hook, register

# ---------------------------------------------------------------------------
# 1. Logging-only hooks (one per call site)
# ---------------------------------------------------------------------------


@hook(HookType.SESSION_PRE_INIT, mode=PluginMode.AUDIT)
async def log_session_pre_init(payload, ctx):
    print("[log] session_pre_init:", payload)


@hook(HookType.SESSION_POST_INIT, mode=PluginMode.AUDIT)
async def log_session_post_init(payload, ctx):
    print("[log] session_post_init:", payload)


@hook(HookType.SESSION_RESET, mode=PluginMode.AUDIT)
async def log_session_reset(payload, ctx):
    print("[log] session_reset:", payload)


@hook(HookType.SESSION_CLEANUP, mode=PluginMode.AUDIT)
async def log_session_cleanup(payload, ctx):
    print("[log] session_cleanup:", payload)


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.AUDIT)
async def log_component_pre_execute(payload, ctx):
    print("[log] component_pre_execute:", payload)


@hook(HookType.COMPONENT_POST_SUCCESS, mode=PluginMode.AUDIT)
async def log_component_post_success(payload, ctx):
    print("[log] component_post_success:", payload)


@hook(HookType.COMPONENT_POST_ERROR, mode=PluginMode.AUDIT)
async def log_component_post_error(payload, ctx):
    print("[log] component_post_error:", payload)


@hook(HookType.GENERATION_PRE_CALL, mode=PluginMode.AUDIT)
async def log_generation_pre_call(payload, ctx):
    print("[log] generation_pre_call:", payload)


@hook(HookType.GENERATION_POST_CALL, mode=PluginMode.AUDIT)
async def log_generation_post_call(payload, ctx):
    print("[log] generation_post_call:", payload)


@hook(HookType.VALIDATION_PRE_CHECK, mode=PluginMode.AUDIT)
async def log_validation_pre_check(payload, ctx):
    print("[log] validation_pre_check:", payload)


@hook(HookType.VALIDATION_POST_CHECK, mode=PluginMode.AUDIT)
async def log_validation_post_check(payload, ctx):
    print("[log] validation_post_check:", payload)


@hook(HookType.SAMPLING_LOOP_START, mode=PluginMode.AUDIT)
async def log_sampling_loop_start(payload, ctx):
    print("[log] sampling_loop_start:", payload)


@hook(HookType.SAMPLING_ITERATION, mode=PluginMode.AUDIT)
async def log_sampling_iteration(payload, ctx):
    print("[log] sampling_iteration:", payload)


@hook(HookType.SAMPLING_REPAIR, mode=PluginMode.AUDIT)
async def log_sampling_repair(payload, ctx):
    print("[log] sampling_repair:", payload)


@hook(HookType.SAMPLING_LOOP_END, mode=PluginMode.AUDIT)
async def log_sampling_loop_end(payload, ctx):
    print("[log] sampling_loop_end:", payload)


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.AUDIT)
async def log_tool_pre_invoke(payload, ctx):
    print("[log] tool_pre_invoke:", payload)


@hook(HookType.TOOL_POST_INVOKE, mode=PluginMode.AUDIT)
async def log_tool_post_invoke(payload, ctx):
    print("[log] tool_post_invoke:", payload)


logging_plugin_set = PluginSet(
    "logging",
    [
        log_session_pre_init,
        log_session_post_init,
        log_session_reset,
        log_session_cleanup,
        log_component_pre_execute,
        log_component_post_success,
        log_component_post_error,
        log_generation_pre_call,
        log_generation_post_call,
        log_validation_pre_check,
        log_validation_post_check,
        log_sampling_loop_start,
        log_sampling_iteration,
        log_sampling_repair,
        log_sampling_loop_end,
        log_tool_pre_invoke,
        log_tool_post_invoke,
    ],
)


# ---------------------------------------------------------------------------
# 2. Sequential hooks (one per call site)
# ---------------------------------------------------------------------------


@hook(HookType.SESSION_PRE_INIT, mode=PluginMode.SEQUENTIAL)
async def seq_session_pre_init(payload, ctx):
    print("[seq] session_pre_init:", payload)


@hook(HookType.SESSION_POST_INIT, mode=PluginMode.SEQUENTIAL)
async def seq_session_post_init(payload, ctx):
    print("[seq] session_post_init:", payload)


@hook(HookType.SESSION_RESET, mode=PluginMode.SEQUENTIAL)
async def seq_session_reset(payload, ctx):
    print("[seq] session_reset:", payload)


@hook(HookType.SESSION_CLEANUP, mode=PluginMode.SEQUENTIAL)
async def seq_session_cleanup(payload, ctx):
    print("[seq] session_cleanup:", payload)


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.SEQUENTIAL)
async def seq_component_pre_execute(payload, ctx):
    print("[seq] component_pre_execute:", payload)


@hook(HookType.COMPONENT_POST_SUCCESS, mode=PluginMode.SEQUENTIAL)
async def seq_component_post_success(payload, ctx):
    print("[seq] component_post_success:", payload)


@hook(HookType.COMPONENT_POST_ERROR, mode=PluginMode.SEQUENTIAL)
async def seq_component_post_error(payload, ctx):
    print("[seq] component_post_error:", payload)


@hook(HookType.GENERATION_PRE_CALL, mode=PluginMode.SEQUENTIAL)
async def seq_generation_pre_call(payload, ctx):
    print("[seq] generation_pre_call:", payload)


@hook(HookType.GENERATION_POST_CALL, mode=PluginMode.SEQUENTIAL)
async def seq_generation_post_call(payload, ctx):
    print("[seq] generation_post_call:", payload)


@hook(HookType.VALIDATION_PRE_CHECK, mode=PluginMode.SEQUENTIAL)
async def seq_validation_pre_check(payload, ctx):
    print("[seq] validation_pre_check:", payload)


@hook(HookType.VALIDATION_POST_CHECK, mode=PluginMode.SEQUENTIAL)
async def seq_validation_post_check(payload, ctx):
    print("[seq] validation_post_check:", payload)


@hook(HookType.SAMPLING_LOOP_START, mode=PluginMode.SEQUENTIAL)
async def seq_sampling_loop_start(payload, ctx):
    print("[seq] sampling_loop_start:", payload)


@hook(HookType.SAMPLING_ITERATION, mode=PluginMode.SEQUENTIAL)
async def seq_sampling_iteration(payload, ctx):
    print("[seq] sampling_iteration:", payload)


@hook(HookType.SAMPLING_REPAIR, mode=PluginMode.SEQUENTIAL)
async def seq_sampling_repair(payload, ctx):
    print("[seq] sampling_repair:", payload)


@hook(HookType.SAMPLING_LOOP_END, mode=PluginMode.SEQUENTIAL)
async def seq_sampling_loop_end(payload, ctx):
    print("[seq] sampling_loop_end:", payload)


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.SEQUENTIAL)
async def seq_tool_pre_invoke(payload, ctx):
    print("[seq] tool_pre_invoke:", payload)


@hook(HookType.TOOL_POST_INVOKE, mode=PluginMode.SEQUENTIAL)
async def seq_tool_post_invoke(payload, ctx):
    print("[seq] tool_post_invoke:", payload)


sequential_plugin_set = PluginSet(
    "sequential",
    [
        seq_session_pre_init,
        seq_session_post_init,
        seq_session_reset,
        seq_session_cleanup,
        seq_component_pre_execute,
        seq_component_post_success,
        seq_component_post_error,
        seq_generation_pre_call,
        seq_generation_post_call,
        seq_validation_pre_check,
        seq_validation_post_check,
        seq_sampling_loop_start,
        seq_sampling_iteration,
        seq_sampling_repair,
        seq_sampling_loop_end,
        seq_tool_pre_invoke,
        seq_tool_post_invoke,
    ],
)


# ---------------------------------------------------------------------------
# 3. Concurrent hooks (one per call site)
# ---------------------------------------------------------------------------


@hook(HookType.SESSION_PRE_INIT, mode=PluginMode.CONCURRENT)
async def concurrent_session_pre_init(payload, ctx):
    print("[concurrent] session_pre_init:", payload)


@hook(HookType.SESSION_POST_INIT, mode=PluginMode.CONCURRENT)
async def concurrent_session_post_init(payload, ctx):
    print("[concurrent] session_post_init:", payload)


@hook(HookType.SESSION_RESET, mode=PluginMode.CONCURRENT)
async def concurrent_session_reset(payload, ctx):
    print("[concurrent] session_reset:", payload)


@hook(HookType.SESSION_CLEANUP, mode=PluginMode.CONCURRENT)
async def concurrent_session_cleanup(payload, ctx):
    print("[concurrent] session_cleanup:", payload)


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.CONCURRENT)
async def concurrent_component_pre_execute(payload, ctx):
    print("[concurrent] component_pre_execute:", payload)


@hook(HookType.COMPONENT_POST_SUCCESS, mode=PluginMode.CONCURRENT)
async def concurrent_component_post_success(payload, ctx):
    print("[concurrent] component_post_success:", payload)


@hook(HookType.COMPONENT_POST_ERROR, mode=PluginMode.CONCURRENT)
async def concurrent_component_post_error(payload, ctx):
    print("[concurrent] component_post_error:", payload)


@hook(HookType.GENERATION_PRE_CALL, mode=PluginMode.CONCURRENT)
async def concurrent_generation_pre_call(payload, ctx):
    print("[concurrent] generation_pre_call:", payload)


@hook(HookType.GENERATION_POST_CALL, mode=PluginMode.CONCURRENT)
async def concurrent_generation_post_call(payload, ctx):
    print("[concurrent] generation_post_call:", payload)


@hook(HookType.VALIDATION_PRE_CHECK, mode=PluginMode.CONCURRENT)
async def concurrent_validation_pre_check(payload, ctx):
    print("[concurrent] validation_pre_check:", payload)


@hook(HookType.VALIDATION_POST_CHECK, mode=PluginMode.CONCURRENT)
async def concurrent_validation_post_check(payload, ctx):
    print("[concurrent] validation_post_check:", payload)


@hook(HookType.SAMPLING_LOOP_START, mode=PluginMode.CONCURRENT)
async def concurrent_sampling_loop_start(payload, ctx):
    print("[concurrent] sampling_loop_start:", payload)


@hook(HookType.SAMPLING_ITERATION, mode=PluginMode.CONCURRENT)
async def concurrent_sampling_iteration(payload, ctx):
    print("[concurrent] sampling_iteration:", payload)


@hook(HookType.SAMPLING_REPAIR, mode=PluginMode.CONCURRENT)
async def concurrent_sampling_repair(payload, ctx):
    print("[concurrent] sampling_repair:", payload)


@hook(HookType.SAMPLING_LOOP_END, mode=PluginMode.CONCURRENT)
async def concurrent_sampling_loop_end(payload, ctx):
    print("[concurrent] sampling_loop_end:", payload)


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.CONCURRENT)
async def concurrent_tool_pre_invoke(payload, ctx):
    print("[concurrent] tool_pre_invoke:", payload)


@hook(HookType.TOOL_POST_INVOKE, mode=PluginMode.CONCURRENT)
async def concurrent_tool_post_invoke(payload, ctx):
    print("[concurrent] tool_post_invoke:", payload)


concurrent_plugin_set = PluginSet(
    "concurrent",
    [
        concurrent_session_pre_init,
        concurrent_session_post_init,
        concurrent_session_reset,
        concurrent_session_cleanup,
        concurrent_component_pre_execute,
        concurrent_component_post_success,
        concurrent_component_post_error,
        concurrent_generation_pre_call,
        concurrent_generation_post_call,
        concurrent_validation_pre_check,
        concurrent_validation_post_check,
        concurrent_sampling_loop_start,
        concurrent_sampling_iteration,
        concurrent_sampling_repair,
        concurrent_sampling_loop_end,
        concurrent_tool_pre_invoke,
        concurrent_tool_post_invoke,
    ],
)

# ---------------------------------------------------------------------------
# 4. Fire-and-forget hooks (one per call site)
# ---------------------------------------------------------------------------


@hook(HookType.SESSION_PRE_INIT, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_session_pre_init(payload, ctx):
    print("[fandf] session_pre_init:", payload)


@hook(HookType.SESSION_POST_INIT, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_session_post_init(payload, ctx):
    print("[fandf] session_post_init:", payload)


@hook(HookType.SESSION_RESET, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_session_reset(payload, ctx):
    print("[fandf] session_reset:", payload)


@hook(HookType.SESSION_CLEANUP, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_session_cleanup(payload, ctx):
    print("[fandf] session_cleanup:", payload)


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_component_pre_execute(payload, ctx):
    print("[fandf] component_pre_execute:", payload)


@hook(HookType.COMPONENT_POST_SUCCESS, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_component_post_success(payload, ctx):
    print("[fandf] component_post_success:", payload)


@hook(HookType.COMPONENT_POST_ERROR, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_component_post_error(payload, ctx):
    print("[fandf] component_post_error:", payload)


@hook(HookType.GENERATION_PRE_CALL, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_generation_pre_call(payload, ctx):
    print("[fandf] generation_pre_call:", payload)


@hook(HookType.GENERATION_POST_CALL, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_generation_post_call(payload, ctx):
    print("[fandf] generation_post_call:", payload)


@hook(HookType.VALIDATION_PRE_CHECK, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_validation_pre_check(payload, ctx):
    print("[fandf] validation_pre_check:", payload)


@hook(HookType.VALIDATION_POST_CHECK, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_validation_post_check(payload, ctx):
    print("[fandf] validation_post_check:", payload)


@hook(HookType.SAMPLING_LOOP_START, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_sampling_loop_start(payload, ctx):
    print("[fandf] sampling_loop_start:", payload)


@hook(HookType.SAMPLING_ITERATION, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_sampling_iteration(payload, ctx):
    print("[fandf] sampling_iteration:", payload)


@hook(HookType.SAMPLING_REPAIR, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_sampling_repair(payload, ctx):
    print("[fandf] sampling_repair:", payload)


@hook(HookType.SAMPLING_LOOP_END, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_sampling_loop_end(payload, ctx):
    print("[fandf] sampling_loop_end:", payload)


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_tool_pre_invoke(payload, ctx):
    print("[fandf] tool_pre_invoke:", payload)


@hook(HookType.TOOL_POST_INVOKE, mode=PluginMode.FIRE_AND_FORGET)
async def fandf_tool_post_invoke(payload, ctx):
    print("[fandf] tool_post_invoke:", payload)


fandf_plugin_set = PluginSet(
    "fandf",
    [
        fandf_session_pre_init,
        fandf_session_post_init,
        fandf_session_reset,
        fandf_session_cleanup,
        fandf_component_pre_execute,
        fandf_component_post_success,
        fandf_component_post_error,
        fandf_generation_pre_call,
        fandf_generation_post_call,
        fandf_validation_pre_check,
        fandf_validation_post_check,
        fandf_sampling_loop_start,
        fandf_sampling_iteration,
        fandf_sampling_repair,
        fandf_sampling_loop_end,
        fandf_tool_pre_invoke,
        fandf_tool_post_invoke,
    ],
)

register(logging_plugin_set)
# register(sequential_plugin_set)
# register(concurrent_plugin_set)
# register(fandf_plugin_set)


@hook(HookType.SESSION_PRE_INIT, mode=PluginMode.AUDIT)
async def log_session_pre_init_two(payload, ctx):
    print("[log] session_pre_init_two:", payload)


register(log_session_pre_init_two)

# register(blocking_plugin_set)
