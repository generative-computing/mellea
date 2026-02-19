"""OpenTelemetry instrumentation for Mellea.

This package provides observability capabilities for Mellea through OpenTelemetry,
enabling tracing and metrics collection for both application-level operations and
backend LLM interactions.

Package Structure:
    - tracing: Distributed tracing with two independent scopes:
        * Application traces (mellea.application): User-facing operations
        * Backend traces (mellea.backend): LLM backend interactions
    - metrics: Metrics collection for counters, histograms, and up-down counters
    - backend_instrumentation: Automatic instrumentation for backend operations

Configuration:
    All telemetry features are opt-in via environment variables:
    
    Tracing:
        - MELLEA_TRACE_APPLICATION: Enable application tracing (default: false)
        - MELLEA_TRACE_BACKEND: Enable backend tracing (default: false)
        - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint for trace export
        - OTEL_SERVICE_NAME: Service name for traces (default: mellea)
    
    Metrics:
        - MELLEA_METRICS_ENABLED: Enable metrics collection (default: false)
        - MELLEA_METRICS_CONSOLE: Print metrics to console (default: false)
        - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint for metric export (optional)
        - OTEL_SERVICE_NAME: Service name for metrics (default: mellea)

Dependencies:
    OpenTelemetry packages are optional. If not installed, telemetry features
    are gracefully disabled. Install with: pip install mellea[telemetry]

Example:
    from mellea.telemetry import trace_application, create_counter
    
    # Trace application operations
    @trace_application("my_operation")
    def my_function():
        pass
    
    # Collect metrics
    counter = create_counter("mellea.requests", unit="1")
    counter.add(1, {"backend": "ollama"})
"""

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
