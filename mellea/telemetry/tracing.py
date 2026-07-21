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
"""

from __future__ import annotations

import asyncio
import os
import warnings
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

from mellea.telemetry._tracing_helpers import (
    _env_true,
    content_capture_enabled,
    get_capture_content_value,
    get_tool_call_attrs,
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


_in_flight_spans: dict[
    str, tuple[Span, Token[Context] | None, asyncio.Task[Any] | None]
] = {}

# reattach_span() entries, keyed by correlation key: the OTel context token plus
# the task that attached it. Released by release_reattached_span() on that task.
_reattached_tokens: dict[str, tuple[Token[Context], asyncio.Task[Any] | None]] = {}


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


def _current_task() -> asyncio.Task[Any] | None:
    """Return the running asyncio task, or None when no loop is running."""
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _safe_detach(
    token: Token[Context] | None, attach_task: asyncio.Task[Any] | None
) -> None:
    """Detach `token`, suppressing only the cross-task detach we expect and understand.

    OTel context tokens are bound to the `contextvars.Context` of the task that
    created them, so detaching from a different task fails — OTel catches the
    `ValueError` and logs it at ERROR as "Failed to detach context". Most spans
    attach and finish on one task and never hit this.

    A cross-task detach is suppressed (skipped, since it would only fail) and
    logged at debug only when the detaching task holds an open `reattach_span`
    scope — the marker that this task knowingly opened a span elsewhere and
    expects the mismatch. Any other cross-task detach is left to run so OTel
    surfaces its ERROR with a traceback to the real origin; a warning is added
    first to name the task mismatch OTel's message omits.

    Example:
        Under `stream_with_chunking` the backend `chat` span attaches in the
        caller task but finishes in the orchestration task that drains the MOT.
        To keep that span's `chat` children nesting correctly, the orchestration
        task re-attaches the streaming span for the duration of the drain
        (`reattach_span` / `release_reattached_span`). The cross-task `chat`
        detach that then happens within that scope is the expected, suppressed
        case.

    Note:
        The reattach scope is an *incomplete* proxy for "expected". It holds for
        streaming because that case both needs sibling-nesting protection (so it
        reattaches) and has an expected cross-task detach. A future case with the
        same open-in-parent / close-in-child shape but no siblings to protect
        would not reattach, so its equally-expected cross-task detach falls
        through to the warn-and-detach path. That is harmless (the detach only
        fails, and the task ends right after, so nothing leaks); the warning is
        the signal that the new use needs its own way to mark the detach expected.

    Args:
        token: The OTel context token returned by the matching `attach`, or
            `None` when attach was skipped — a no-op.
        attach_task: The task that performed the `attach`, or None if it was
            attached outside any running task.
    """
    if token is None:
        return
    current = _current_task()
    if attach_task is not None and current is not attach_task:
        from mellea.core.utils import MelleaLogger

        if any(task is current for _, task in _reattached_tokens.values()):
            MelleaLogger.get_logger().debug(
                "Skipped expected cross-task OTel context detach within a "
                "reattached-span scope."
            )
            return
        MelleaLogger.get_logger().warning(
            "Detaching an OTel context token across asyncio tasks; the span's "
            "attach and detach ran on different tasks. OTel will log the failure."
        )
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
        action_class_name: Optional `mellea.action_type` attribute (chat).
        num_actions: Optional `mellea.num_actions` attribute (batch).
        has_format: Optional `mellea.has_format` attribute (whether structured
            output was requested).
        format_type: Optional `mellea.format_type` attribute (the structured
            output class name, when `has_format` is True).
        tool_calls_enabled: Optional `mellea.tool_calls_enabled` attribute.
        attach_context: Whether to attach the span as the ambient OTel context.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    from mellea.core.base import GenerationMetadata

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

    token = _attach_span_context(span, attach=attach_context)
    _in_flight_spans[generation_id] = (span, token, _current_task())
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
    span, token, attach_task = entry
    try:
        if gen is not None:
            set_request_attrs(span, gen, operation)
            set_response_attrs(span, gen)
        set_usage_attrs(span, usage)
        if mot is not None and gen is not None:
            set_mellea_attrs(span, mot, gen)
    finally:
        _safe_detach(token, attach_task)
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
    span, token, attach_task = entry
    try:
        if gen is not None:
            set_request_attrs(span, gen, operation)
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
        span.set_attribute("error.type", type(exception).__name__)
    finally:
        _safe_detach(token, attach_task)
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
    _in_flight_spans[key] = (span, token, _current_task())
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
    span, token, attach_task = entry
    try:
        if extra_attributes:
            for k, v in extra_attributes.items():
                set_attribute_safe(span, k, v)
    finally:
        _safe_detach(token, attach_task)
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
    span, token, attach_task = entry
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
        _safe_detach(token, attach_task)
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

    Carries construction-time attributes (`mellea.session_id`,
    `mellea.backend`, `mellea.model_id`, `mellea.context_type`). Stashed
    under a derived key so it doesn't collide with the long-lived
    `session` span when both share a `session_id`.

    Args:
        session_id: Session UUID. The in-flight key is derived from this.
        backend: Backend identifier (e.g. `"ollama"`); stamped as `mellea.backend`.
        model_id: Resolved model id string.
        context_type: Context class name (e.g. `"SimpleContext"`).

    Returns:
        The span, or `None` if tracing is disabled.
    """
    return _start_application_span(
        "start_session",
        session_id + _SESSION_STARTUP_KEY_SUFFIX,
        {
            "mellea.session_id": session_id,
            "mellea.backend": backend,
            "mellea.model_id": model_id,
            "mellea.context_type": context_type,
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
        session_id: Session UUID, used as the correlation key and stamped
            as `mellea.session_id`.
        context_type: Context class name.
        backend: Backend identifier (e.g. `"ollama"`); stamped as
            `mellea.backend` when provided.

    Returns:
        The span, or `None` if tracing is disabled.
    """
    return _start_application_span(
        "session",
        session_id,
        {
            "mellea.session_id": session_id,
            "mellea.context_type": context_type,
            "mellea.backend": backend,
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
            "mellea.action_type": action_class_name,
            "mellea.has_requirements": has_requirements,
            "mellea.has_strategy": has_strategy,
            "mellea.strategy_type": strategy_type,
            "mellea.has_format": has_format,
            "mellea.tool_calls": tool_calls,
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

    `mellea.response` is recorded (truncated) only when content capture is enabled;
    `mellea.response_length` is always recorded (a non-content metric).

    Args:
        action_id: Correlation key from the matching open call.
        num_generate_logs: `mellea.num_generate_logs`.
        sampling_success: `mellea.sampling_success` (set when a strategy ran).
        response_text: Raw response text. Recorded as `mellea.response` only
            when content tracing is enabled.
        response_length: `mellea.response_length` (always safe; ungated).
    """
    _finish_application_span_success(
        action_id,
        extra_attributes={
            "mellea.num_generate_logs": num_generate_logs,
            "mellea.sampling_success": sampling_success,
            "mellea.response": get_capture_content_value(response_text),
            "mellea.response_length": response_length,
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
    """Open the `stream_with_chunking` span for one orchestration run.

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
        "stream_with_chunking",
        streaming_id,
        {
            "mellea.has_requirements": has_requirements,
            "mellea.requirement_count": requirement_count,
            "mellea.chunking_strategy": chunking_strategy,
        },
        attach_context=attach_context,
    )


def add_streaming_event(
    streaming_id: str, *, event_name: str, attributes: dict[str, Any]
) -> None:
    """Add an OTel span event to the in-flight `stream_with_chunking` span.

    Leaves the span in `_in_flight_spans` for a later `finish_streaming_span_*`
    call to close.

    Args:
        streaming_id: Correlation key from the matching open call.
        event_name: Span-event name.
        attributes: Span-event attributes; `None` values are skipped.
    """
    entry = _in_flight_spans.get(streaming_id)
    if entry is None:
        return
    span = entry[0]
    filtered = {k: v for k, v in attributes.items() if v is not None}
    span.add_event(event_name, filtered)


def reattach_span(key: str) -> None:
    """Make the in-flight span `key` the current task's ambient context.

    Spans opened by later work on this task then parent under it. Paired with
    `release_reattached_span()`, which must run on the same task. No-op when the
    span is not in flight. See `_safe_detach` for how this scope is used to
    classify the cross-task detach it enables.

    Args:
        key: Correlation key of an in-flight span (the key it was stashed under).
    """
    entry = _in_flight_spans.get(key)
    if entry is None:
        return
    span = entry[0]
    token = otel_context.attach(trace.set_span_in_context(span))
    _reattached_tokens[key] = (token, _current_task())


def release_reattached_span(key: str) -> None:
    """Release a reattached span from a matching `reattach_span()` call.

    Must run on the same task that called `reattach_span()`. No-op when no token
    is stored.

    Args:
        key: Correlation key from the matching `reattach_span()` call.
    """
    entry = _reattached_tokens.pop(key, None)
    if entry is not None:
        token, _ = entry
        otel_context.detach(token)


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
    """End the `stream_with_chunking` span, recording its outcome.

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
        full_text_length: Accumulated text length at orchestrator exit.
    """
    extra_attributes = {
        "mellea.full_text_length": full_text_length,
        "gen_ai.request.model": model,
        "gen_ai.provider.name": provider,
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


__all__ = [
    "get_application_tracer",
    "get_backend_tracer",
    "is_content_tracing_enabled",
    "is_tracing_enabled",
    "start_backend_span",
]
