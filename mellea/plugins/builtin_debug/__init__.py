"""Built-in debug plugins for Mellea.

Provides pre-built plugins for common debugging tasks:
- Generation pipeline tracing (requests, responses, latency, tokens)
- Sampling strategy diagnostics (iterations, validation, repair, results)

Examples:
    Enable generation tracing:

        from mellea.plugins.builtin_debug.generation import (
            log_generation_pre_call,
            log_generation_post_call,
        )
        from mellea.plugins import register

        register([log_generation_pre_call, log_generation_post_call])

    Enable sampling diagnostics:

        from mellea.plugins.builtin_debug.sampling import (
            log_sampling_loop_start,
            log_sampling_iteration,
            log_sampling_repair,
            log_sampling_loop_end,
        )

        register([
            log_sampling_loop_start,
            log_sampling_iteration,
            log_sampling_repair,
            log_sampling_loop_end,
        ])
"""

from __future__ import annotations

from .generation import log_generation_post_call, log_generation_pre_call
from .sampling import (
    log_sampling_iteration,
    log_sampling_loop_end,
    log_sampling_loop_start,
    log_sampling_repair,
)
from .validation import log_validation_post_check, log_validation_pre_check

__all__ = [
    "log_generation_post_call",
    "log_generation_pre_call",
    "log_sampling_iteration",
    "log_sampling_loop_end",
    "log_sampling_loop_start",
    "log_sampling_repair",
    "log_validation_post_check",
    "log_validation_pre_check",
]
