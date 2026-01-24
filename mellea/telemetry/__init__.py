"""OpenTelemetry instrumentation for Mellea.

This module provides two independent trace scopes:
1. Application Trace (mellea.application) - User-facing operations
2. Backend Trace (mellea.backend) - LLM backend interactions

Configuration via environment variables:
- MELLEA_TRACE_APPLICATION: Enable/disable application tracing (default: false)
- MELLEA_TRACE_BACKEND: Enable/disable backend tracing (default: false)
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint for trace export
- OTEL_SERVICE_NAME: Service name for traces (default: mellea)
"""

import os
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Configuration from environment variables
_TRACE_APPLICATION_ENABLED = os.getenv("MELLEA_TRACE_APPLICATION", "false").lower() in (
    "true",
    "1",
    "yes",
)
_TRACE_BACKEND_ENABLED = os.getenv("MELLEA_TRACE_BACKEND", "false").lower() in (
    "true",
    "1",
    "yes",
)
_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "mellea")
_CONSOLE_EXPORT = os.getenv("MELLEA_TRACE_CONSOLE", "false").lower() in (
    "true",
    "1",
    "yes",
)


def _setup_tracer_provider() -> TracerProvider:
    """Set up the global tracer provider with OTLP exporter if configured."""
    resource = Resource.create({"service.name": _SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint is configured
    if _OTLP_ENDPOINT:
        otlp_exporter = OTLPSpanExporter(endpoint=_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Add console exporter for debugging if enabled
    # Note: Console exporter may cause harmless errors during test cleanup
    if _CONSOLE_EXPORT:
        try:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
        except Exception:
            # Silently ignore console exporter setup failures
            pass

    trace.set_tracer_provider(provider)
    return provider


# Initialize tracer provider if any tracing is enabled
_tracer_provider = None
if _TRACE_APPLICATION_ENABLED or _TRACE_BACKEND_ENABLED:
    _tracer_provider = _setup_tracer_provider()

# Create separate tracers for application and backend
_application_tracer = trace.get_tracer("mellea.application", "0.3.0")
_backend_tracer = trace.get_tracer("mellea.backend", "0.3.0")


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
    if _TRACE_APPLICATION_ENABLED:
        with _application_tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, str(value))
            yield span
    else:
        yield None


@contextmanager
def trace_backend(name: str, **attributes: Any):
    """Create a backend trace span if backend tracing is enabled.

    Args:
        name: Name of the span
        **attributes: Additional attributes to add to the span

    Yields:
        The span object if tracing is enabled, otherwise a no-op context manager
    """
    if _TRACE_BACKEND_ENABLED:
        with _backend_tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, str(value))
            yield span
    else:
        yield None


def set_span_attribute(span: Any, key: str, value: Any) -> None:
    """Set an attribute on a span if the span is not None.

    Args:
        span: The span object (may be None if tracing is disabled)
        key: Attribute key
        value: Attribute value
    """
    if span is not None and value is not None:
        span.set_attribute(key, str(value))


def set_span_error(span: Any, exception: Exception) -> None:
    """Record an exception on a span if the span is not None.

    Args:
        span: The span object (may be None if tracing is disabled)
        exception: The exception to record
    """
    if span is not None:
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


__all__ = [
    "is_application_tracing_enabled",
    "is_backend_tracing_enabled",
    "set_span_attribute",
    "set_span_error",
    "trace_application",
    "trace_backend",
]
