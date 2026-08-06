---
id: tracing
title: "mellea.telemetry.tracing"
sidebar_label: "tracing"
sidebar_position: 6
description: "OpenTelemetry tracing instrumentation for Mellea."
# diataxis: reference
---

Source: [`mellea/telemetry/tracing.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py) at commit `a535fc6345a0`.

OpenTelemetry tracing instrumentation for Mellea.

Declared exports (`__all__`): `get_application_tracer`, `get_backend_tracer`, `is_content_tracing_enabled`, `is_tracing_enabled`, `start_backend_span`

## `is_tracing_enabled()`

*function* — [line 167](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L167)

`is_tracing_enabled() -> bool`

Check if tracing is enabled.

## `is_content_tracing_enabled()`

*function* — [line 203](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L203)

`is_content_tracing_enabled() -> bool`

Check if content capture is enabled.

## `get_application_tracer()`

*function* — [line 216](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L216)

`get_application_tracer() -> Any`

Return the application tracer.

## `get_backend_tracer()`

*function* — [line 226](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L226)

`get_backend_tracer() -> Any`

Return the backend tracer.

## `start_backend_span()`

*function* — [line 333](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L333)

`start_backend_span(operation: str, generation_id: str, *, model: str | None, provider: str | None, action_class_name: str | None = None, num_actions: int | None = None, has_format: bool | None = None, format_type: str | None = None, tool_calls_enabled: bool | None = None, attach_context: bool = True) -> Span | None`

Open a backend span, activate it as the current OTel context, and stash both under `generation_id`.

## `finish_backend_span_success()`

*function* — [line 400](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L400)

`finish_backend_span_success(generation_id: str, *, operation: str, usage: dict[str, Any] | None, mot: Any | None, gen: Any | None) -> None`

Add response-side attrs and end the in-flight backend span.

## `finish_backend_span_error()`

*function* — [line 437](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L437)

`finish_backend_span_error(generation_id: str, *, operation: str, exception: BaseException, gen: Any | None = None) -> None`

Set ERROR status, record the exception, and end the in-flight span.

## `start_session_startup_span()`

*function* — [line 563](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L563)

`start_session_startup_span(session_id: str, *, backend: str | None, model_id: str | None, context_type: str | None) -> Span | None`

Open the `start_session` span around backend construction.

## `finish_session_startup_span()`

*function* — [line 598](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L598)

`finish_session_startup_span(session_id: str, *, exception: BaseException | None = None) -> bool`

End the nested `start_session` span if one is in flight.

## `start_session_span()`

*function* — [line 621](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L621)

`start_session_span(session_id: str, *, context_type: str | None, backend: str | None = None) -> Span | None`

Open the long-lived `session` span over a session's lifetime.

## `finish_session_span()`

*function* — [line 647](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L647)

`finish_session_span(session_id: str, *, exception: BaseException | None = None) -> None`

End the long-lived `session` span.

## `start_action_span()`

*function* — [line 662](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L662)

`start_action_span(action_id: str, *, action_class_name: str | None, has_requirements: bool | None, has_strategy: bool | None, strategy_type: str | None, has_format: bool | None, tool_calls: bool | None, attach_context: bool = True) -> Span | None`

Open the `action` span for a single component execution.

## `finish_action_span_success()`

*function* — [line 703](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L703)

`finish_action_span_success(action_id: str, *, num_generate_logs: int | None = None, sampling_success: bool | None = None, response_text: str | None = None, response_length: int | None = None) -> None`

End the action span with response-side attributes.

## `finish_action_span_error()`

*function* — [line 735](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L735)

`finish_action_span_error(action_id: str, *, exception: BaseException | None) -> None`

End the action span with ERROR status.

## `start_tool_span()`

*function* — [line 748](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L748)

`start_tool_span(tool_invocation_id: str, model_tool_call: Any, *, is_control_flow: bool, attach_context: bool = True) -> Span | None`

Open the `execute_tool` span for a single tool invocation.

## `finish_tool_span_success()`

*function* — [line 776](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L776)

`finish_tool_span_success(tool_invocation_id: str, *, execution_time_ms: int, result: Any | None) -> None`

End the tool span with success status and response-side attributes.

## `finish_tool_span_error()`

*function* — [line 800](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L800)

`finish_tool_span_error(tool_invocation_id: str, *, execution_time_ms: int, exception: BaseException | None) -> None`

End the tool span with ERROR status, recording the exception.

## `start_streaming_span()`

*function* — [line 821](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L821)

`start_streaming_span(streaming_id: str, *, has_requirements: bool | None, requirement_count: int | None, chunking_strategy: str | None, attach_context: bool = True) -> Span | None`

Open the `stream_with_chunking` span for one orchestration run.

## `add_span_event()`

*function* — [line 853](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L853)

`add_span_event(key: str, *, event_name: str, attributes: dict[str, Any]) -> None`

Add an OTel span event to any in-flight application span.

## `reattach_span()`

*function* — [line 871](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L871)

`reattach_span(key: str) -> None`

Make the in-flight span `key` the current task's ambient context.

## `release_reattached_span()`

*function* — [line 890](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L890)

`release_reattached_span(key: str) -> None`

Release a reattached span from a matching `reattach_span()` call.

## `finish_streaming_span()`

*function* — [line 905](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L905)

`finish_streaming_span(streaming_id: str, *, success: bool, failure_reason: str | None = None, exception: BaseException | None = None, model: str | None = None, provider: str | None = None, full_text_length: int | None = None) -> None`

End the `stream_with_chunking` span, recording its outcome.

## `start_sampling_span()`

*function* — [line 950](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L950)

`start_sampling_span(sampling_id: str, *, strategy_type: str | None, loop_budget: int | None, requirement_count: int | None, attach_context: bool = True) -> Span | None`

Open the `sampling` span for a single sampling loop.

## `finish_sampling_span()`

*function* — [line 985](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L985)

`finish_sampling_span(sampling_id: str, *, success: bool, iterations_used: int | None = None, failure_reason: str | None = None, exception: BaseException | None = None) -> None`

End the `sampling` span.

## `start_validation_span()`

*function* — [line 1019](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L1019)

`start_validation_span(validation_id: str, *, requirement_count: int | None, attach_context: bool = True) -> Span | None`

Open the `validation` span for a single requirement-validation batch.

## `finish_validation_span()`

*function* — [line 1040](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing.py#L1040)

`finish_validation_span(validation_id: str, *, all_validations_passed: bool | None = None, passed_count: int | None = None, failed_count: int | None = None, failure_reasons: list[str] | None = None, exception: BaseException | None = None) -> None`

End the `validation` span.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
