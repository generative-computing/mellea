"""OpenTelemetry instrumentation for Mellea.

This module provides two independent trace scopes:
1. Application Trace (mellea.application) - User-facing operations
2. Backend Trace (mellea.backend) - LLM backend interactions

Follows OpenTelemetry Gen-AI semantic conventions:
https://opentelemetry.io/docs/specs/semconv/gen-ai/

Configuration via environment variables:
- MELLEA_TRACE_APPLICATION: Enable/disable application tracing (default: false)
- MELLEA_TRACE_BACKEND: Enable/disable backend tracing (default: false)
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint for trace export
- OTEL_SERVICE_NAME: Service name for traces (default: mellea)
"""

import os
from contextlib import contextmanager
from typing import Any

# Try to import OpenTelemetry, but make it optional
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.semconv.trace import SpanAttributes

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    # Provide dummy types for type hints
    trace = None  # type: ignore
    SpanAttributes = None  # type: ignore

# Configuration from environment variables
# Disable tracing if OpenTelemetry is not available
_TRACE_APPLICATION_ENABLED = _OTEL_AVAILABLE and os.getenv(
    "MELLEA_TRACE_APPLICATION", "false"
).lower() in ("true", "1", "yes")
_TRACE_BACKEND_ENABLED = _OTEL_AVAILABLE and os.getenv(
    "MELLEA_TRACE_BACKEND", "false"
).lower() in ("true", "1", "yes")
_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "mellea")
_CONSOLE_EXPORT = os.getenv("MELLEA_TRACE_CONSOLE", "false").lower() in (
    "true",
    "1",
    "yes",
)


def _setup_tracer_provider():
    """Set up the global tracer provider with OTLP exporter if configured."""
    if not _OTEL_AVAILABLE:
        return None

    resource = Resource.create({"service.name": _SERVICE_NAME})  # type: ignore
    provider = TracerProvider(resource=resource)  # type: ignore

    # Add OTLP exporter if endpoint is configured
    if _OTLP_ENDPOINT:
        otlp_exporter = OTLPSpanExporter(endpoint=_OTLP_ENDPOINT)  # type: ignore
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))  # type: ignore

    # Add console exporter for debugging if enabled
    # Note: Console exporter may cause harmless errors during test cleanup
    if _CONSOLE_EXPORT:
        try:
            console_exporter = ConsoleSpanExporter()  # type: ignore
            provider.add_span_processor(BatchSpanProcessor(console_exporter))  # type: ignore
        except Exception:
            # Silently ignore console exporter setup failures
            pass

    trace.set_tracer_provider(provider)  # type: ignore
    return provider


# Initialize tracer provider if any tracing is enabled
_tracer_provider = None
_application_tracer = None
_backend_tracer = None

if _OTEL_AVAILABLE and (_TRACE_APPLICATION_ENABLED or _TRACE_BACKEND_ENABLED):
    _tracer_provider = _setup_tracer_provider()
    # Create separate tracers for application and backend
    _application_tracer = trace.get_tracer("mellea.application", "0.3.0")  # type: ignore
    _backend_tracer = trace.get_tracer("mellea.backend", "0.3.0")  # type: ignore


def is_application_tracing_enabled() -> bool:
    """Check if application tracing is enabled."""
    return _TRACE_APPLICATION_ENABLED


def is_backend_tracing_enabled() -> bool:
    """Check if backend tracing is enabled."""
    return _TRACE_BACKEND_ENABLED


@contextmanager
def trace_application(name: str, **attributes: Any):
    """Create an application trace span if application tracing is enabled.

    Args:
        name: Name of the span
        **attributes: Additional attributes to add to the span

    Yields:
        The span object if tracing is enabled, otherwise a no-op context manager
    """
    if _TRACE_APPLICATION_ENABLED and _application_tracer is not None:
        with _application_tracer.start_as_current_span(name) as span:  # type: ignore
            for key, value in attributes.items():
                if value is not None:
                    _set_attribute_safe(span, key, value)
            yield span
    else:
        yield None


@contextmanager
def trace_backend(name: str, **attributes: Any):
    """Create a backend trace span if backend tracing is enabled.

    Follows Gen-AI semantic conventions for LLM operations.

    Args:
        name: Name of the span
        **attributes: Additional attributes to add to the span

    Yields:
        The span object if tracing is enabled, otherwise a no-op context manager
    """
    if _TRACE_BACKEND_ENABLED and _backend_tracer is not None:
        with _backend_tracer.start_as_current_span(name) as span:  # type: ignore
            # Set Gen-AI operation type
            span.set_attribute("gen_ai.operation.name", name)

            for key, value in attributes.items():
                if value is not None:
                    _set_attribute_safe(span, key, value)
            yield span
    else:
        yield None


def start_backend_span(name: str, **attributes: Any):
    """Start a backend trace span without auto-closing (for async operations).

    Use this when you need to manually control span lifecycle, such as for
    async operations where the span should remain open until post-processing.

    Args:
        name: Name of the span
        **attributes: Additional attributes to add to the span

    Returns:
        The span object if tracing is enabled, otherwise None
    """
    if _TRACE_BACKEND_ENABLED and _backend_tracer is not None:
        span = _backend_tracer.start_span(name)  # type: ignore
        # Set Gen-AI operation type
        span.set_attribute("gen_ai.operation.name", name)

        for key, value in attributes.items():
            if value is not None:
                _set_attribute_safe(span, key, value)
        return span
    return None


def end_backend_span(span: Any) -> None:
    """End a backend trace span.

    Args:
        span: The span object to end
    """
    if span is not None:
        span.end()


def _set_attribute_safe(span: Any, key: str, value: Any) -> None:
    """Set an attribute on a span, handling type conversions.

    Args:
        span: The span object
        key: Attribute key
        value: Attribute value (will be converted to appropriate type)
    """
    if value is None:
        return

    # Handle different value types according to OpenTelemetry spec
    if isinstance(value, bool):
        span.set_attribute(key, value)
    elif isinstance(value, int | float):
        span.set_attribute(key, value)
    elif isinstance(value, str):
        span.set_attribute(key, value)
    elif isinstance(value, list | tuple):
        # Convert to list of strings
        span.set_attribute(key, [str(v) for v in value])
    else:
        # Convert other types to string
        span.set_attribute(key, str(value))


def set_span_attribute(span: Any, key: str, value: Any) -> None:
    """Set an attribute on a span if the span is not None.

    Args:
        span: The span object (may be None if tracing is disabled)
        key: Attribute key
        value: Attribute value
    """
    if span is not None and value is not None:
        _set_attribute_safe(span, key, value)


def set_span_error(span: Any, exception: Exception) -> None:
    """Record an exception on a span if the span is not None.

    Args:
        span: The span object (may be None if tracing is disabled)
        exception: The exception to record
    """
    if span is not None and _OTEL_AVAILABLE:
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))  # type: ignore


__all__ = [
    "is_application_tracing_enabled",
    "is_backend_tracing_enabled",
    "set_span_attribute",
    "set_span_error",
    "trace_application",
    "trace_backend",
]
