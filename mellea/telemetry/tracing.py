# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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

Consumption boundary:
    This is a public API for external consumers. Code outside
    `mellea/telemetry/` opens spans via hook plugins (`tracing_plugins.py`), not
    by calling these functions. Exceptions are limited to spans opened from sync
    code, and are documented at their call site.
"""

from __future__ import annotations

import os
import warnings
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.context import Context, Token

    # NOTE: this is the OpenTelemetry tracing span, distinct from the unrelated
    # `mellea.core.Span` alias (the Component | CBlock | ModelOutputThunk union).
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

from mellea.telemetry._tracing_helpers import (
    _env_true,
    content_capture_enabled,
    get_capture_content_value,
    get_tool_call_attrs,
    normalize_provider_name,
    set_attribute_safe,
    set_conversation_id,
    set_mellea_attrs,
    set_request_attrs,
    set_response_attrs,
    set_usage_attrs,
)

_tracer_provider: Any = None
_application_tracer: Any = None
_backend_tracer: Any = None
_tracing_enabled: bool = False
_plugins_registered: bool = False  # Plugin registry is process-global; register once.

_REMOTE_PROVIDERS = frozenset({"openai", "ollama", "watsonx", "litellm"})


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
        except ValueError as e:
            warnings.warn(
                f"{plugin_cls.__name__} already registered: {e}",
                UserWarning,
                stacklevel=2,
            )
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
    return _OTEL_AVAILABLE and content_capture_enabled()


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


_in_flight_spans: dict[str, tuple[Span, Token[Context] | None]] = {}


def _attach_span_context(span: Span, *, attach: bool) -> Token[Context] | None:
    """Attach `span` as the current OTel context, unless `attach` is False.

    When `attach` is False the span is left detached and `None` is returned,
    signalling paired detach sites to skip detaching.

    Args:
        span: The span to activate as the ambient OTel context.
        attach: Whether to perform the attach at all.

    Returns:
        The OTel context token to pass to a later detach, or `None` when attach
        was skipped.
    """
    if not attach:
        return None
    return otel_context.attach(trace.set_span_in_context(span))


def _detach_token(token: Token[Context] | None) -> None:
    """Detach an OTel context token, or no-op when `token` is `None`.

    Args:
        token: The token returned by the matching `attach`, or `None` when
            attach was skipped.
    """
    if token is None:
        return
    otel_context.detach(token)


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
    streaming: bool | None = None,
    attach_context: bool = True,
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
        action_class_name: Component class name being generated from (chat).
        num_actions: Number of actions in the batch call (batch).
        has_format: Whether structured output was requested; emits
            `gen_ai.output.type="json"` when True.
        format_type: Structured-output class name, when `has_format` is True.
        tool_calls_enabled: Whether tool calling is enabled for the call.
        streaming: Whether streaming was requested; emits `gen_ai.request.stream`
            when True (unset/False is left off, which semconv reads as non-streaming).
        attach_context: Whether to attach the span as the ambient OTel context.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    from mellea.core.base import GenerationMetadata

    tracer = get_backend_tracer()
    if tracer is None:
        return None

    # Span name is "{operation} {model}" per Gen-AI semconv; operation alone when model unknown.
    span_name = f"{operation} {model}" if model else operation
    kind = (
        trace.SpanKind.CLIENT
        if provider in _REMOTE_PROVIDERS
        else trace.SpanKind.INTERNAL
    )
    span = tracer.start_span(span_name, kind=kind)

    gen = GenerationMetadata(model=model, provider=provider)
    set_request_attrs(span, gen, operation)
    if action_class_name is not None:
        span.set_attribute("mellea.component.type", action_class_name)
    if num_actions is not None:
        span.set_attribute("mellea.request.num_actions", num_actions)
    if has_format:
        span.set_attribute("gen_ai.output.type", "json")
    if format_type is not None:
        span.set_attribute("mellea.request.format_type", format_type)
    if streaming:
        span.set_attribute("gen_ai.request.stream", True)
    if tool_calls_enabled is not None:
        span.set_attribute("mellea.action.tool_calls", tool_calls_enabled)
    set_conversation_id(span)

    token = _attach_span_context(span, attach=attach_context)
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
    entry = _in_flight_spans.pop(generation_id, None)
    if entry is None:
        return
    span, token = entry
    try:
        if gen is not None:
            set_request_attrs(span, gen, operation)
            set_response_attrs(span, gen)
        set_usage_attrs(span, usage)
        if mot is not None:
            set_mellea_attrs(span, mot)
    finally:
        _detach_token(token)
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
        _detach_token(token)
        span.end()


def _start_application_span(
    name: str, key: str, attributes: dict[str, Any], *, attach_context: bool = True
) -> Span | None:
    """Open an application span, attach it to the OTel context, and stash by key.

    Args:
        name: Span name.
        key: Correlation key for the in-flight stash.
        attributes: Initial attributes; `None` values are skipped.
        attach_context: Whether to attach the span as the ambient OTel context.

    Returns:
        The span, or `None` if the application tracer is unavailable.
    """
    tracer = get_application_tracer()
    if tracer is None:
        return None

    span = tracer.start_span(name)
    for k, v in attributes.items():
        if v is not None:
            set_attribute_safe(span, k, v)

    token = _attach_span_context(span, attach=attach_context)
    _in_flight_spans[key] = (span, token)
    return span


def _finish_application_span_success(
    key: str, *, extra_attributes: dict[str, Any] | None = None
) -> None:
    """End an in-flight application span with default (OK) status.

    Detaches the OTel context token before ending so subsequent work parents
    correctly. Tokens are task-affine — callers must arrange for detach to
    happen on the same task that attached.

    Args:
        key: Correlation key from the matching open call.
        extra_attributes: Optional response-side attributes; `None` values are skipped.
    """
    entry = _in_flight_spans.pop(key, None)
    if entry is None:
        return
    span, token = entry
    try:
        if extra_attributes:
            for k, v in extra_attributes.items():
                set_attribute_safe(span, k, v)
    finally:
        _detach_token(token)
        span.end()


def _finish_application_span_error(
    key: str,
    *,
    extra_attributes: dict[str, Any] | None = None,
    exception: BaseException | None = None,
    description: str | None = None,
) -> None:
    """End an in-flight application span with ERROR status.

    Records `exception` when given (status + recorded exception + `error.type`);
    otherwise sets ERROR status from `description` with no recorded exception.
    Detaches the OTel context token before ending.

    Args:
        key: Correlation key from the matching open call.
        extra_attributes: Optional response-side attributes; `None` values are skipped.
        exception: The exception to record, when one was raised.
        description: ERROR-status description used when `exception` is `None`.
    """
    entry = _in_flight_spans.pop(key, None)
    if entry is None:
        return
    span, token = entry
    try:
        if extra_attributes:
            for k, v in extra_attributes.items():
                set_attribute_safe(span, k, v)
        if exception is not None:
            span.record_exception(exception)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
            span.set_attribute("error.type", type(exception).__name__)
        else:
            span.set_status(trace.Status(trace.StatusCode.ERROR, description or ""))
    finally:
        _detach_token(token)
        span.end()


_SESSION_STARTUP_KEY_SUFFIX = ":startup"


def start_session_startup_span(
    session_id: str,
    *,
    backend: str | None,
    model_id: str | None,
    context_type: str | None,
) -> Span | None:
    """Open the `start_session` span around backend construction.

    Stashed under a derived key so it doesn't collide with the long-lived
    `session` span when both share a `session_id`.

    Args:
        session_id: Session UUID. The in-flight key is derived from this.
        backend: Requested backend name (e.g. `"ollama"`, `"hf"`), before
            resolution to a provider id.
        model_id: Resolved model id string.
        context_type: Context class name (e.g. `"SimpleContext"`).

    Returns:
        The span, or `None` if tracing is disabled.
    """
    return _start_application_span(
        "start_session",
        session_id + _SESSION_STARTUP_KEY_SUFFIX,
        {
            "mellea.session.id": session_id,
            "mellea.session.backend_name": backend,
            "gen_ai.request.model": model_id,
            "mellea.session.context_type": context_type,
        },
    )


def finish_session_startup_span(
    session_id: str, *, exception: BaseException | None = None
) -> bool:
    """End the nested `start_session` span if one is in flight.

    Args:
        session_id: Session UUID from the matching open call. The in-flight
            key is derived from this.
        exception: If provided, mark the span ERROR.

    Returns:
        True if a child span was open and was finished; False if no-op.
    """
    key = session_id + _SESSION_STARTUP_KEY_SUFFIX
    if key not in _in_flight_spans:
        return False
    if exception is not None:
        _finish_application_span_error(key, exception=exception)
    else:
        _finish_application_span_success(key)
    return True


def start_session_span(
    session_id: str, *, context_type: str | None, backend: str | None = None
) -> Span | None:
    """Open the long-lived `session` span over a session's lifetime.

    Args:
        session_id: Session UUID, used as the correlation key.
        context_type: Context class name.
        backend: Resolved provider id (e.g. `"ollama"`), normalized to its
            `gen_ai.provider.name` value when provided.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    return _start_application_span(
        "session",
        session_id,
        {
            "mellea.session.id": session_id,
            "mellea.session.context_type": context_type,
            "gen_ai.provider.name": normalize_provider_name(backend),
        },
    )


def finish_session_span(
    session_id: str, *, exception: BaseException | None = None
) -> None:
    """End the long-lived `session` span.

    Args:
        session_id: Correlation key from the matching open call.
        exception: If provided, mark the span ERROR.
    """
    if exception is not None:
        _finish_application_span_error(session_id, exception=exception)
    else:
        _finish_application_span_success(session_id)


def start_action_span(
    action_id: str,
    *,
    action_class_name: str | None,
    has_requirements: bool | None,
    has_strategy: bool | None,
    strategy_type: str | None,
    has_format: bool | None,
    tool_calls: bool | None,
    attach_context: bool = True,
) -> Span | None:
    """Open the `action` span for a single component execution.

    Args:
        action_id: UUID correlating this component execution across hooks.
        action_class_name: Class name of the component being executed.
        has_requirements: Whether requirements were supplied.
        has_strategy: Whether a sampling strategy was supplied.
        strategy_type: Sampling strategy class name when present.
        has_format: Whether a structured-output format was supplied.
        tool_calls: Whether tool calling is enabled.
        attach_context: Whether to attach the span as the ambient OTel context.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    return _start_application_span(
        "action",
        action_id,
        {
            "mellea.component.type": action_class_name,
            "mellea.action.has_requirements": has_requirements,
            "mellea.action.has_strategy": has_strategy,
            "mellea.sampling.strategy_type": strategy_type,
            "mellea.action.has_format": has_format,
            "mellea.action.tool_calls": tool_calls,
        },
        attach_context=attach_context,
    )


def finish_action_span_success(
    action_id: str,
    *,
    num_generate_logs: int | None = None,
    sampling_success: bool | None = None,
    response_text: str | None = None,
    response_length: int | None = None,
) -> None:
    """End the action span with response-side attributes.

    The response text is recorded (truncated) only when content capture is
    enabled; its length is always recorded (a non-content metric).

    Args:
        action_id: Correlation key from the matching open call.
        num_generate_logs: Number of generate logs the run accumulated.
        sampling_success: Sampling outcome, set when a strategy ran.
        response_text: Raw response text. Recorded only when content tracing
            is enabled.
        response_length: Response length; always safe to record (ungated).
    """
    _finish_application_span_success(
        action_id,
        extra_attributes={
            "mellea.action.num_generate_logs": num_generate_logs,
            "mellea.sampling.success": sampling_success,
            "mellea.action.response": get_capture_content_value(response_text),
            "mellea.action.response_length": response_length,
        },
    )


def finish_action_span_error(
    action_id: str, *, exception: BaseException | None
) -> None:
    """End the action span with ERROR status.

    Args:
        action_id: Correlation key from the matching open call.
        exception: The exception that ended the action, or `None` to set ERROR
            status without a recorded exception.
    """
    _finish_application_span_error(action_id, exception=exception)


def start_tool_span(
    tool_invocation_id: str,
    model_tool_call: Any,
    *,
    is_control_flow: bool,
    attach_context: bool = True,
) -> Span | None:
    """Open the `execute_tool` span for a single tool invocation.

    Args:
        tool_invocation_id: UUID correlating this invocation across the pre/post hooks.
        model_tool_call: The `ModelToolCall` being executed.
        is_control_flow: Whether this tool is framework control flow.
        attach_context: Whether to attach the span as the ambient OTel context.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    attrs = get_tool_call_attrs(model_tool_call)
    attrs["mellea.tool.is_control_flow"] = is_control_flow
    return _start_application_span(
        f"execute_tool {attrs['gen_ai.tool.name']}",
        tool_invocation_id,
        attrs,
        attach_context=attach_context,
    )


def finish_tool_span_success(
    tool_invocation_id: str, *, execution_time_ms: int, result: Any | None
) -> None:
    """End the tool span with success status and response-side attributes.

    `gen_ai.tool.call.result` is recorded (truncated) only when content capture
    is enabled.

    Args:
        tool_invocation_id: Correlation key from the matching open call.
        execution_time_ms: Wall-clock tool execution time.
        result: The tool's return value. Recorded as `gen_ai.tool.call.result`
            only when content tracing is enabled.
    """
    _finish_application_span_success(
        tool_invocation_id,
        extra_attributes={
            "mellea.tool.status": "success",
            "mellea.tool.execution_time_ms": execution_time_ms,
            "gen_ai.tool.call.result": get_capture_content_value(result),
        },
    )


def finish_tool_span_error(
    tool_invocation_id: str, *, execution_time_ms: int, exception: BaseException | None
) -> None:
    """End the tool span with ERROR status, recording the exception.

    Args:
        tool_invocation_id: Correlation key from the matching open call.
        execution_time_ms: Wall-clock tool execution time.
        exception: The exception raised by the tool, or `None` to set ERROR
            status without a recorded exception.
    """
    _finish_application_span_error(
        tool_invocation_id,
        extra_attributes={
            "mellea.tool.status": "failure",
            "mellea.tool.execution_time_ms": execution_time_ms,
        },
        exception=exception,
    )


def start_streaming_span(
    streaming_id: str,
    *,
    has_requirements: bool | None,
    requirement_count: int | None,
    chunking_strategy: str | None,
    attach_context: bool = True,
) -> Span | None:
    """Open the `stream` span for one streaming run.

    Args:
        streaming_id: UUID correlating this streaming run across hooks.
        has_requirements: Whether requirements were supplied.
        requirement_count: Number of requirements supplied.
        chunking_strategy: ChunkingStrategy class name.
        attach_context: Whether to attach the span as the ambient OTel context.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    return _start_application_span(
        "stream",
        streaming_id,
        {
            "mellea.streaming.has_requirements": has_requirements,
            "mellea.streaming.requirement_count": requirement_count,
            "mellea.streaming.chunking_strategy": chunking_strategy,
        },
        attach_context=attach_context,
    )


def add_span_event(key: str, *, event_name: str, attributes: dict[str, Any]) -> None:
    """Add an OTel span event to any in-flight application span.

    Leaves the span in `_in_flight_spans` for a later `finish_*` call to close.

    Args:
        key: Correlation key from the matching open call.
        event_name: Span-event name.
        attributes: Span-event attributes; `None` values are skipped.
    """
    entry = _in_flight_spans.get(key)
    if entry is None:
        return
    span = entry[0]
    filtered = {k: v for k, v in attributes.items() if v is not None}
    span.add_event(event_name, filtered)


def finish_streaming_span(
    streaming_id: str,
    *,
    success: bool,
    failure_reason: str | None = None,
    exception: BaseException | None = None,
    model: str | None = None,
    provider: str | None = None,
    full_text_length: int | None = None,
) -> None:
    """End the `stream` span, recording its outcome.

    Sets OK status on success. On failure, marks the span ERROR: with the
    exception recorded when one is given, otherwise with `failure_reason` and
    no recorded exception.

    Args:
        streaming_id: Correlation key from the matching open call.
        success: `True` only on a clean completion.
        failure_reason: Human-readable ERROR-status description, used when
            `success` is `False` and no `exception` is given.
        exception: The exception raised by the orchestrator, when one was.
        model: Model identifier, when known.
        provider: Provider name, when known.
        full_text_length: Length of the validated-and-emitted text at stream exit.
    """
    extra_attributes = {
        "mellea.streaming.full_text_length": full_text_length,
        "gen_ai.request.model": model,
        "gen_ai.provider.name": normalize_provider_name(provider),
    }

    if success:
        _finish_application_span_success(
            streaming_id, extra_attributes=extra_attributes
        )
    else:
        _finish_application_span_error(
            streaming_id,
            extra_attributes=extra_attributes,
            exception=exception,
            description=failure_reason,
        )


def start_sampling_span(
    sampling_id: str,
    *,
    strategy_type: str | None,
    loop_budget: int | None,
    requirement_count: int | None,
    attach_context: bool = True,
) -> Span | None:
    """Open the `sampling` span for a single sampling loop.

    Iterations and repairs are recorded as span events on this span (see
    `add_span_event`) rather than as child spans.

    Args:
        sampling_id: UUID correlating this loop across the sampling hooks.
        strategy_type: Sampling strategy class name.
        loop_budget: Maximum iterations per subsample.
        requirement_count: Number of requirements validated each iteration.
        attach_context: Whether to attach the span as the ambient OTel context.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    return _start_application_span(
        "sampling",
        sampling_id,
        {
            "mellea.sampling.strategy_type": strategy_type,
            "mellea.sampling.loop_budget": loop_budget,
            "mellea.sampling.requirement_count": requirement_count,
        },
        attach_context=attach_context,
    )


def finish_sampling_span(
    sampling_id: str,
    *,
    success: bool,
    iterations_used: int | None = None,
    failure_reason: str | None = None,
    exception: BaseException | None = None,
) -> None:
    """End the `sampling` span.

    Records the outcome attributes, or ERROR status with the exception when the
    loop raised.

    Args:
        sampling_id: Correlation key from the matching open call.
        success: `True` if at least one attempt passed all requirements.
        iterations_used: Total iterations that completed across subsamples.
        failure_reason: Reason recorded when `success` is `False`.
        exception: The exception that ended the loop, when one was raised.
    """
    if exception is None:
        _finish_application_span_success(
            sampling_id,
            extra_attributes={
                "mellea.sampling.success": success,
                "mellea.sampling.iterations_used": iterations_used,
                "mellea.sampling.failure_reason": failure_reason,
            },
        )
    else:
        _finish_application_span_error(sampling_id, exception=exception)


def start_validation_span(
    validation_id: str, *, requirement_count: int | None, attach_context: bool = True
) -> Span | None:
    """Open the `validation` span for a single requirement-validation batch.

    Args:
        validation_id: UUID correlating the pre/post validation hooks.
        requirement_count: Number of requirements being validated.
        attach_context: Whether to attach the span as the ambient OTel context.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    return _start_application_span(
        "validation",
        validation_id,
        {"mellea.validation.requirement_count": requirement_count},
        attach_context=attach_context,
    )


def finish_validation_span(
    validation_id: str,
    *,
    all_validations_passed: bool | None = None,
    passed_count: int | None = None,
    failed_count: int | None = None,
    failure_reasons: list[str] | None = None,
    exception: BaseException | None = None,
) -> None:
    """End the `validation` span.

    Records the outcome attributes, or ERROR status with the exception when the
    check raised. Failure reasons are recorded only when content capture is
    enabled, since a requirement's reason can echo model output.

    Args:
        validation_id: Correlation key from the matching open call.
        all_validations_passed: Whether every requirement passed.
        passed_count: Number of requirements that passed.
        failed_count: Number of requirements that failed.
        failure_reasons: One reason per failing requirement. Recorded only when
            content tracing is enabled.
        exception: The exception that ended validation, when one was raised.
    """
    if exception is None:
        reasons = (
            [get_capture_content_value(r) for r in failure_reasons]
            if failure_reasons and content_capture_enabled()
            else None
        )
        _finish_application_span_success(
            validation_id,
            extra_attributes={
                "mellea.validation.passed": all_validations_passed,
                "mellea.validation.passed_count": passed_count,
                "mellea.validation.failed_count": failed_count,
                "mellea.validation.failure_reasons": reasons,
            },
        )
    else:
        _finish_application_span_error(validation_id, exception=exception)


# Child phase spans are stashed under a key derived from the invocation id so
# they can't collide with the parent's own `_in_flight_spans` entry (keyed by
# the bare invocation id) or with each other across phases of the same
# invocation.
_ADAPTER_FUNCTION_PHASE_KEY_INFIX = ":phase:"


def _adapter_function_phase_key(invocation_id: str, phase: str) -> str:
    return f"{invocation_id}{_ADAPTER_FUNCTION_PHASE_KEY_INFIX}{phase}"


def start_adapter_function_span(
    invocation_id: str,
    *,
    name: str,
    revision: str | None,
    binding_type: str,
    adapter_type: str,
) -> Span | None:
    """Open the `adapter_function` parent span for one adapter-function invocation.

    On the `mellea.backend` tracer: adapter/model lifecycle work is a backend
    concern, not a user-facing operation (see
    `docs/docs/observability/tracing.md`).

    Never attached as the ambient OTel context, unlike every other
    `start_*_span` helper — deliberately, not as an oversight.
    `ADAPTER_FUNCTION_INVOCATION_START`/`_PHASE_START` fire from synchronous
    `AdapterMixin.adapter_scope` code via `_run_async_in_thread`, which runs
    each hook as an independent task on a shared background event loop, seeded
    from a *fresh* `contextvars.copy_context()` snapshot of the calling thread.
    Mutations inside one hook's task (like an ambient-context attach) never
    leak back to the calling thread and are invisible to the next dispatch.
    Ambient attach/detach across these calls therefore cannot establish a
    parent/child edge — `start_adapter_function_phase_span` parents explicitly
    via `trace.set_span_in_context` using the span object looked up by
    `invocation_id`. Exposing an `attach_context` parameter here would offer a
    setting that breaks whenever used, so there isn't one.

    Args:
        invocation_id: Correlation key for the matching `finish_adapter_function_span` call.
        name: Adapter function name (e.g. `"answerability"`).
        revision: Catalog revision of the adapter, or `None` if unpinned.
        binding_type: Weight-binding reality the adapter is running under (e.g.
            `"local_file"`, `"embedded"`, `"server_mediated"`).
        adapter_type: Adapter mechanism (e.g. `"lora"`, `"alora"`).

    Returns:
        The span, or `None` if tracing is disabled.
    """
    tracer = get_backend_tracer()
    if tracer is None:
        return None

    span = tracer.start_span("adapter_function")
    set_attribute_safe(span, "mellea.adapter_function.name", name)
    set_attribute_safe(span, "mellea.adapter_function.revision", revision)
    set_attribute_safe(span, "mellea.adapter_function.binding_type", binding_type)
    set_attribute_safe(span, "mellea.adapter_function.adapter_type", adapter_type)

    _in_flight_spans[invocation_id] = (span, None)
    return span


def finish_adapter_function_span(
    invocation_id: str, *, outcome: str, exception: BaseException | None
) -> None:
    """End the `adapter_function` span, recording its outcome.

    Defensively closes any `adapter_function.<phase>` child span still open
    under this invocation first — a phase that raised fires
    `adapter_function_phase_start` but never its own
    `adapter_function_phase_complete` (that hook's contract is success-only),
    so without this the child span would never close and the in-flight
    registry would never drain. The dangling child receives the invocation
    exception unless it is `deactivate` and the invocation body also failed:
    that exception remains primary, so copying it onto the deactivation child
    would misattribute the failure.

    Args:
        invocation_id: Correlation key from the matching `start_adapter_function_span` call.
        outcome: `"success"`, `"schema_error"`, or `"error"`.
        exception: The exception raised during the invocation, or `None` on success.
    """
    # `list(...)` snapshots the keys in one call before filtering, so a
    # concurrent insert from another thread (e.g. a different invocation's
    # sync-dispatched hook, on the shared `_run_async_in_thread` background
    # loop) can't trigger a "dictionary changed size during iteration" error
    # here — the first site in this module to iterate `_in_flight_spans`
    # rather than do a keyed lookup.
    prefix = f"{invocation_id}{_ADAPTER_FUNCTION_PHASE_KEY_INFIX}"
    for key in [k for k in list(_in_flight_spans) if k.startswith(prefix)]:
        entry = _in_flight_spans.pop(key, None)
        if entry is None:
            continue
        phase_span, _phase_token = entry
        phase = key.removeprefix(prefix)
        deactivation_failure_noted = (
            phase == "deactivate"
            and exception is not None
            and any(
                note.startswith("Adapter deactivation also failed:")
                for note in getattr(exception, "__notes__", ())
            )
        )
        if exception is not None:
            if deactivation_failure_noted:
                phase_span.set_status(
                    trace.Status(trace.StatusCode.ERROR, "Phase did not complete")
                )
            else:
                phase_span.record_exception(exception)
                phase_span.set_status(
                    trace.Status(trace.StatusCode.ERROR, str(exception))
                )
                phase_span.set_attribute("error.type", type(exception).__name__)
        phase_span.end()

    entry = _in_flight_spans.pop(invocation_id, None)
    if entry is None:
        return
    span, _token = entry
    set_attribute_safe(span, "mellea.adapter_function.outcome", outcome)
    if exception is not None:
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
        span.set_attribute("error.type", type(exception).__name__)
    span.end()


def start_adapter_function_phase_span(
    invocation_id: str, phase: str, *, revision: str | None = None
) -> Span | None:
    """Open an `adapter_function.<phase>` child span, explicitly parented under the invocation span.

    Parents via `trace.set_span_in_context` on the `adapter_function` span
    looked up by `invocation_id`, rather than via ambient context — see
    `start_adapter_function_span`'s docstring for why ambient attach can't
    establish this edge for hooks fired via `_run_async_in_thread`. Explicit
    parenting works the same on every Python version; no `_CONTEXT_ATTACH_SUPPORTED`
    gating applies here. A missing parent (invocation not in flight, or
    tracing was enabled only after the invocation started) falls back to
    whatever's ambient — the phase span still opens; it just can't nest.

    Args:
        invocation_id: Correlation key of the enclosing `adapter_function` span.
        phase: Lifecycle phase name (e.g. `"activate"`, `"deactivate"`).
        revision: Catalog revision of the adapter, or `None` if unpinned. Recorded
            directly on the phase span as well as the parent.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    tracer = get_backend_tracer()
    if tracer is None:
        return None

    parent_entry = _in_flight_spans.get(invocation_id)
    parent_context = (
        trace.set_span_in_context(parent_entry[0]) if parent_entry is not None else None
    )
    span = tracer.start_span(f"adapter_function.{phase}", context=parent_context)
    set_attribute_safe(span, "mellea.adapter_function.phase", phase)
    set_attribute_safe(span, "mellea.adapter_function.revision", revision)

    _in_flight_spans[_adapter_function_phase_key(invocation_id, phase)] = (span, None)
    return span


def finish_adapter_function_phase_span(invocation_id: str, phase: str) -> None:
    """End an `adapter_function.<phase>` child span successfully.

    A no-op if the phase span isn't in flight — e.g. it was already closed
    defensively by `finish_adapter_function_span` because the phase raised.

    Args:
        invocation_id: Correlation key of the enclosing `adapter_function` span.
        phase: Lifecycle phase name.
    """
    entry = _in_flight_spans.pop(
        _adapter_function_phase_key(invocation_id, phase), None
    )
    if entry is None:
        return
    span, _token = entry
    span.end()


__all__ = [
    "get_application_tracer",
    "get_backend_tracer",
    "is_content_tracing_enabled",
    "is_tracing_enabled",
    "start_backend_span",
]
