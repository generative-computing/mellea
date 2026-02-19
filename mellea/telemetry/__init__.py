"""OpenTelemetry instrumentation for Mellea."""

from .metrics import (
    create_counter,
    create_histogram,
    create_up_down_counter,
    is_metrics_enabled,
)
from .tracing import (
    end_backend_span,
    is_application_tracing_enabled,
    is_backend_tracing_enabled,
    set_span_attribute,
    set_span_error,
    start_backend_span,
    trace_application,
    trace_backend,
)

__all__ = [
    "create_counter",
    "create_histogram",
    "create_up_down_counter",
    "end_backend_span",
    "is_application_tracing_enabled",
    "is_backend_tracing_enabled",
    "is_metrics_enabled",
    "set_span_attribute",
    "set_span_error",
    "start_backend_span",
    "trace_application",
    "trace_backend",
]
