"""OpenTelemetry tracing instrumentation for Mellea.

Provides distributed tracing with two independent tracer scopes:

1. Application Trace (`mellea.application`) - User-facing operations
2. Backend Trace (`mellea.backend`) - LLM backend interactions

Follows OpenTelemetry Gen-AI semantic conventions:
https://opentelemetry.io/docs/specs/semconv/gen-ai/

Configuration via environment variables:

- `MELLEA_TRACES_ENABLED`: Enable tracing (default: `false`).
- `MELLEA_TRACES_OTLP`: Enable OTLP span exporter (default: `false`).
- `MELLEA_TRACES_CONSOLE`: Print spans to console (default: `false`).
- `MELLEA_TRACES_CONTENT`: Capture prompt/response content on spans (default:
  `false`). Content may include PII; enable only in controlled environments.
  Also recognised: `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`
  (OTel standard).
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`: Trace-specific OTLP endpoint (optional).
- `OTEL_EXPORTER_OTLP_ENDPOINT`: General OTLP endpoint (fallback).
- `OTEL_SERVICE_NAME`: Service name for traces (default: `mellea`).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.context import Context, Token
    from opentelemetry.trace import Span

try:
    from opentelemetry import context as otel_context, trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    trace = None  # type: ignore
    otel_context = None  # type: ignore


def _env_true(name: str) -> bool:
    """Return True if `name` is set to a truthy value (1/true/yes)."""
    return os.getenv(name, "false").lower() in ("true", "1", "yes")


_tracer_provider: Any = None
_application_tracer: Any = None
_backend_tracer: Any = None
_tracing_enabled: bool = False
_plugins_registered: bool = False  # Plugin registry is process-global; register once.


def _setup_tracer_provider() -> Any:
    """Set up the global TracerProvider with configured exporters.

    Reads endpoint, exporter, and service-name env vars at call time.

    Returns:
        TracerProvider instance, or None if OpenTelemetry is not available.
    """
    if not _OTEL_AVAILABLE:
        return None

    service_name = os.getenv("OTEL_SERVICE_NAME", "mellea")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    otlp_enabled = _env_true("MELLEA_TRACES_OTLP")
    if otlp_enabled:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )
        if endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except Exception as e:
                warnings.warn(
                    f"Failed to initialize OTLP trace exporter: {e}. "
                    "Spans will not be exported via OTLP.",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            warnings.warn(
                "OTLP trace exporter is enabled (MELLEA_TRACES_OTLP=true) but no "
                "endpoint is configured. Set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or "
                "OTEL_EXPORTER_OTLP_ENDPOINT to export spans.",
                UserWarning,
                stacklevel=3,
            )

    if _env_true("MELLEA_TRACES_CONSOLE"):
        try:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
        except Exception as e:
            warnings.warn(
                f"Failed to initialize console span exporter: {e}. "
                "Spans will not be printed to console.",
                UserWarning,
                stacklevel=3,
            )

    trace.set_tracer_provider(provider)
    return provider


def _register_tracing_plugins() -> None:
    """Register backend tracing plugins on the global plugin registry.

    Idempotent via `_plugins_registered` so test state resets are safe.
    """
    global _plugins_registered
    if _plugins_registered:
        return

    from mellea.plugins.registry import _HAS_PLUGIN_FRAMEWORK, register

    if not _HAS_PLUGIN_FRAMEWORK:
        warnings.warn(
            "Tracing is enabled but the plugin framework is not installed. "
            "Backend spans will not be emitted automatically. "
            "Install with: pip install mellea[telemetry]",
            UserWarning,
            stacklevel=2,
        )
        return

    from mellea.telemetry.tracing_plugins import _TRACING_PLUGIN_CLASSES

    for plugin_cls in _TRACING_PLUGIN_CLASSES:
        try:
            register(plugin_cls())
        except ValueError:
            # Already registered in a previous invocation (e.g. after a
            # state reset in tests). Silent — registry remains correct.
            pass
    _plugins_registered = True


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled.

    Returns:
        True if `MELLEA_TRACES_ENABLED` is truthy AND OpenTelemetry is installed.
    """
    return _tracing_enabled


def _setup_tracing() -> None:
    """Initialise the tracer provider, tracers, and register plugins."""
    global _tracer_provider, _application_tracer, _backend_tracer, _tracing_enabled

    _tracing_enabled = False
    _tracer_provider = None
    _application_tracer = None
    _backend_tracer = None
    if not (_OTEL_AVAILABLE and _env_true("MELLEA_TRACES_ENABLED")):
        return

    _tracer_provider = _setup_tracer_provider()
    if _tracer_provider is None:
        return

    mellea_version = version("mellea")
    _application_tracer = _tracer_provider.get_tracer(
        "mellea.application", mellea_version
    )
    _backend_tracer = _tracer_provider.get_tracer("mellea.backend", mellea_version)
    _tracing_enabled = True
    _register_tracing_plugins()


_setup_tracing()


def is_content_tracing_enabled() -> bool:
    """Check if content capture is enabled.

    Content capture records prompt and response text on spans and may contain
    PII; enable only in controlled environments.

    Returns:
        True if enabled via `MELLEA_TRACES_CONTENT` or
        `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.
    """
    if not _OTEL_AVAILABLE:
        return False
    return _env_true("MELLEA_TRACES_CONTENT") or _env_true(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
    )


def get_application_tracer() -> Any:
    """Return the application tracer.

    Returns:
        Tracer instance for application-level spans, or None if tracing is
        disabled or OpenTelemetry is not available.
    """
    return _application_tracer


def get_backend_tracer() -> Any:
    """Return the backend tracer.

    Returns:
        Tracer instance for backend-level spans, or None if tracing is
        disabled or OpenTelemetry is not available.
    """
    return _backend_tracer


_in_flight_spans: dict[str, tuple[Span, Token[Context]]] = {}


def start_backend_span(
    operation: str,
    generation_id: str,
    *,
    model: str | None,
    provider: str | None,
    action_class_name: str | None = None,
    num_actions: int | None = None,
    has_format: bool | None = None,
    format_type: str | None = None,
    tool_calls_enabled: bool | None = None,
) -> Span | None:
    """Open a backend span, activate it as the current OTel context, and stash both under `generation_id`.

    The span is also attached as the current OTel context so nested
    OTel-instrumented work (HTTP clients, framework wrappers, etc.) parents
    under it. Activation propagates to asyncio tasks spawned after this
    call: each new task snapshots the current context at creation time.

    Args:
        operation: Span name (`"chat"` or `"text_completion"`).
        generation_id: Correlation key for the matching finish call.
        model: Model identifier, or `None` if not yet known (chat path
            populates this in post_processing).
        provider: Provider name, or `None` if not yet known.
        action_class_name: Optional `mellea.action_type` attribute (chat).
        num_actions: Optional `mellea.num_actions` attribute (batch).
        has_format: Optional `mellea.has_format` attribute (whether structured
            output was requested).
        format_type: Optional `mellea.format_type` attribute (the structured
            output class name, when `has_format` is True).
        tool_calls_enabled: Optional `mellea.tool_calls_enabled` attribute.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    from mellea.core.base import GenerationMetadata
    from mellea.telemetry._tracing_setters import set_conversation_id, set_request_attrs

    tracer = get_backend_tracer()
    if tracer is None:
        return None

    span = tracer.start_span(operation)

    gen = GenerationMetadata(model=model, provider=provider)
    set_request_attrs(span, gen, operation)
    if action_class_name is not None:
        span.set_attribute("mellea.action_type", action_class_name)
    if num_actions is not None:
        span.set_attribute("mellea.num_actions", num_actions)
    if has_format is not None:
        span.set_attribute("mellea.has_format", has_format)
        if has_format:
            span.set_attribute("gen_ai.output.type", "json_schema")
    if format_type is not None:
        span.set_attribute("mellea.format_type", format_type)
    if tool_calls_enabled is not None:
        span.set_attribute("mellea.tool_calls_enabled", tool_calls_enabled)
    set_conversation_id(span)

    token = otel_context.attach(trace.set_span_in_context(span))
    _in_flight_spans[generation_id] = (span, token)
    return span


def finish_backend_span_success(
    generation_id: str,
    *,
    operation: str,
    usage: dict[str, Any] | None,
    mot: Any | None,
    gen: Any | None,
) -> None:
    """Add response-side attrs and end the in-flight backend span.

    Refreshes request-side attrs from `gen` first, since chat-path backends
    populate `model`/`provider` on the MOT only after the API call
    returns.

    Args:
        generation_id: Correlation key from the matching pre-call.
        operation: Span name used to refresh request attrs.
        usage: Aggregate token-usage dict (OpenAI shape).
        mot: The fully-computed `ModelOutputThunk`, or `None`.
        gen: The `GenerationMetadata` from the MOT, or `None`.
    """
    from mellea.telemetry._tracing_setters import (
        set_mellea_attrs,
        set_request_attrs,
        set_response_attrs,
        set_usage_attrs,
    )

    entry = _in_flight_spans.pop(generation_id, None)
    if entry is None:
        return
    span, token = entry
    try:
        if gen is not None:
            set_request_attrs(span, gen, operation)
            set_response_attrs(span, gen)
        set_usage_attrs(span, usage)
        if mot is not None and gen is not None:
            set_mellea_attrs(span, mot, gen)
    finally:
        otel_context.detach(token)
        span.end()


def finish_backend_span_error(
    generation_id: str,
    *,
    operation: str,
    exception: BaseException,
    gen: Any | None = None,
) -> None:
    """Set ERROR status, record the exception, and end the in-flight span.

    Args:
        generation_id: Correlation key from the matching pre-call.
        operation: Span name used to refresh request attrs (chat path may
            have late-populated model/provider on the MOT before the error).
        exception: The exception raised by the backend.
        gen: Optional `GenerationMetadata` for refreshing request attrs.
    """
    from mellea.telemetry._tracing_setters import set_request_attrs

    entry = _in_flight_spans.pop(generation_id, None)
    if entry is None:
        return
    span, token = entry
    try:
        if gen is not None:
            set_request_attrs(span, gen, operation)
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
        span.set_attribute("error.type", type(exception).__name__)
    finally:
        otel_context.detach(token)
        span.end()


@contextmanager
def trace_application(name: str, **attributes: Any) -> Generator[Any, None, None]:
    """Create an application trace span if application tracing is enabled.

    Args:
        name: Name of the span.
        **attributes: Additional attributes to add to the span.

    Yields:
        The span object if tracing is enabled, otherwise `None`.
    """
    tracer = get_application_tracer()
    if tracer is not None:
        with tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                if value is not None:
                    set_span_attribute(span, key, value)
            yield span
    else:
        yield None


def _set_attribute_safe(span: Any, key: str, value: Any) -> None:
    """Set an attribute on a span, handling type conversions.

    Args:
        span: The span object.
        key: Attribute key.
        value: Attribute value (will be converted to an OTel-compatible type).
    """
    if value is None:
        return

    if isinstance(value, bool):
        span.set_attribute(key, value)
    elif isinstance(value, int | float):
        span.set_attribute(key, value)
    elif isinstance(value, str):
        span.set_attribute(key, value)
    elif isinstance(value, list | tuple):
        span.set_attribute(key, [str(v) for v in value])
    else:
        span.set_attribute(key, str(value))


def set_span_attribute(span: Any, key: str, value: Any) -> None:
    """Set an attribute on a span if the span is not None.

    Args:
        span: The span object (may be None if tracing is disabled).
        key: Attribute key.
        value: Attribute value.
    """
    if span is not None and value is not None:
        _set_attribute_safe(span, key, value)


def set_span_error(span: Any, exception: BaseException) -> None:
    """Record an exception on a span if the span is not None.

    Args:
        span: The span object (may be None if tracing is disabled).
        exception: The exception to record.
    """
    if span is not None and _OTEL_AVAILABLE:
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


def set_span_status_error(span: Any, description: str) -> None:
    """Mark a span as ERROR without recording a phantom exception event.

    Use this for validation failures and other non-exception error conditions
    where the span should be marked failed but no exception was actually raised.
    Calling `set_span_error` in these cases would create a misleading recorded
    exception event in OTEL traces.

    Args:
        span: The span object (may be None if tracing is disabled)
        description: Human-readable reason for the failure.
    """
    if span is not None and _OTEL_AVAILABLE:
        span.set_status(trace.Status(trace.StatusCode.ERROR, description))  # type: ignore


__all__ = [
    "get_application_tracer",
    "get_backend_tracer",
    "is_content_tracing_enabled",
    "is_tracing_enabled",
    "set_span_attribute",
    "set_span_error",
    "set_span_status_error",
    "start_backend_span",
    "trace_application",
]
