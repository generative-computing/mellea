"""OpenTelemetry instrumentation for Mellea.

This package provides observability capabilities for Mellea through OpenTelemetry,
enabling tracing, metrics, and logging for both application-level operations and
backend LLM interactions.

Package Structure:
    - tracing: Distributed tracing with two independent scopes:
        * Application traces (mellea.application): User-facing operations
        * Backend traces (mellea.backend): LLM backend interactions
    - metrics: Metrics collection for counters, histograms, and up-down counters
    - logging: Log export via OTLP
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
        - MELLEA_METRICS_OTLP: Enable OTLP metrics exporter (default: false)
        - MELLEA_METRICS_PROMETHEUS: Enable Prometheus metric reader (default: false)
        - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint for metric export (optional)
        - OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: Metrics-specific OTLP endpoint (optional)
        - OTEL_METRIC_EXPORT_INTERVAL: Export interval in milliseconds (default: 60000)
        - OTEL_SERVICE_NAME: Service name for metrics (default: mellea)
        - MELLEA_PRICING_FILE: Path to a JSON file with custom model pricing overrides (optional)

    Logging:
        - MELLEA_LOG_OTLP: Enable OTLP log export (default: false)
        - OTEL_EXPORTER_OTLP_LOG_ENDPOINT: Logs-specific endpoint (optional)
        - OTEL_EXPORTER_OTLP_ENDPOINT: General OTLP endpoint (fallback)
        - OTEL_SERVICE_NAME: Service name for logs (default: mellea)

Dependencies:
    OpenTelemetry packages are optional. If not installed, telemetry features
    are gracefully disabled. Install with: pip install mellea[telemetry]

Example:
    from mellea.telemetry import trace_application, create_counter, get_otlp_log_handler
    import logging

    # Trace application operations
    @trace_application("my_operation")
    def my_function():
        pass

    # Collect metrics
    counter = create_counter("mellea.requests", unit="1")
    counter.add(1, {"backend": "ollama"})

    # Export logs via OTLP
    logger = logging.getLogger("my_app")
    handler = get_otlp_log_handler()
    if handler:
        logger.addHandler(handler)
"""

from .context import (
    MelleaContextFilter,
    async_with_context,
    generate_request_id,
    get_current_context,
    get_model_id,
    get_request_id,
    get_sampling_iteration,
    get_session_id,
    with_context,
)
from .logging import get_otlp_log_handler
from .metrics import (
    create_counter,
    create_histogram,
    create_up_down_counter,
    is_metrics_enabled,
    record_cost,
    record_error,
    record_request_duration,
    record_requirement_check,
    record_requirement_failure,
    record_sampling_attempt,
    record_sampling_outcome,
    record_token_usage_metrics,
    record_tool_call,
    record_ttfb,
)
from .pricing import is_pricing_enabled
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
    "MelleaContextFilter",
    "async_with_context",
    "create_counter",
    "create_histogram",
    "create_up_down_counter",
    "end_backend_span",
    "generate_request_id",
    "get_current_context",
    "get_model_id",
    "get_otlp_log_handler",
    "get_request_id",
    "get_sampling_iteration",
    "get_session_id",
    "is_application_tracing_enabled",
    "is_backend_tracing_enabled",
    "is_metrics_enabled",
    "is_pricing_enabled",
    "record_cost",
    "record_error",
    "record_request_duration",
    "record_requirement_check",
    "record_requirement_failure",
    "record_sampling_attempt",
    "record_sampling_outcome",
    "record_token_usage_metrics",
    "record_tool_call",
    "record_ttfb",
    "set_span_attribute",
    "set_span_error",
    "start_backend_span",
    "trace_application",
    "trace_backend",
    "with_context",
]
