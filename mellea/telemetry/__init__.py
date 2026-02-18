"""OpenTelemetry instrumentation for Mellea."""

# Import tracing functions
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
    "end_backend_span",
    "is_application_tracing_enabled",
    "is_backend_tracing_enabled",
    "set_span_attribute",
    "set_span_error",
    "start_backend_span",
    "trace_application",
    "trace_backend",
]

