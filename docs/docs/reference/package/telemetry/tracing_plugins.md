---
id: tracing_plugins
title: "mellea.telemetry.tracing_plugins"
sidebar_label: "tracing_plugins"
sidebar_position: 7
description: "Tracing plugins for emitting OpenTelemetry spans via hooks."
# diataxis: reference
---

Source: [`mellea/telemetry/tracing_plugins.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py) at commit `a535fc6345a0`.

Tracing plugins for emitting OpenTelemetry spans via hooks.

## `BackendTracingPlugin`

*class* — [line 70](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L70) (`Plugin`)

Emits Gen-AI semconv backend spans for every LLM generation.

Methods (defined on this class; inherited members not listed):

- `on_pre_call(payload: GenerationPreCallPayload, context: dict[str, Any]) -> None` *(async)* — [line 85](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L85)
  Start a backend chat span for this generation.
- `on_post_call(payload: GenerationPostCallPayload, context: dict[str, Any]) -> None` *(async)* — [line 108](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L108)
  Add usage / mellea attrs and end the chat span.
- `on_error(payload: GenerationErrorPayload, context: dict[str, Any]) -> None` *(async)* — [line 123](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L123)
  Set ERROR status and end the chat span.
- `on_batch_pre_call(payload: GenerationBatchPreCallPayload, context: dict[str, Any]) -> None` *(async)* — [line 143](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L143)
  Start a backend text_completion span for the whole batch.
- `on_batch_post_call(payload: GenerationBatchPostCallPayload, context: dict[str, Any]) -> None` *(async)* — [line 165](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L165)
  Add aggregate usage attrs and end the batch span.
- `on_batch_error(payload: GenerationBatchErrorPayload, context: dict[str, Any]) -> None` *(async)* — [line 182](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L182)
  Set ERROR status and end the batch span.

## `ComponentTracingPlugin`

*class* — [line 197](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L197) (`Plugin`)

Emits application-level spans tracking component execution.

Methods (defined on this class; inherited members not listed):

- `on_component_pre_execute(payload: ComponentPreExecutePayload, context: dict[str, Any]) -> None` *(async)* — [line 209](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L209)
  Open the action span for this component execution.
- `on_component_post_success(payload: ComponentPostSuccessPayload, context: dict[str, Any]) -> None` *(async)* — [line 231](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L231)
  End the action span with response-side attributes.
- `on_component_post_error(payload: ComponentPostErrorPayload, context: dict[str, Any]) -> None` *(async)* — [line 271](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L271)
  End the action span with ERROR status.

## `StreamingTracingPlugin`

*class* — [line 282](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L282) (`Plugin`)

Emits the `stream_with_chunking` application span.

Methods (defined on this class; inherited members not listed):

- `on_streaming_start(payload: StreamingStartPayload, context: dict[str, Any]) -> None` *(async)* — [line 298](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L298)
  Open the stream_with_chunking span for this orchestrator invocation.
- `on_streaming_orchestration_start(payload: StreamingOrchestrationStartPayload, context: dict[str, Any]) -> None` *(async)* — [line 315](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L315)
  Re-attach the streaming span as the orchestration task's ambient context.
- `on_streaming_orchestration_end(payload: StreamingOrchestrationEndPayload, context: dict[str, Any]) -> None` *(async)* — [line 326](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L326)
  Detach the streaming span re-attached on the orchestration task.
- `on_streaming_event(payload: StreamingEventPayload, context: dict[str, Any]) -> None` *(async)* — [line 337](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L337)
  Record a span event for one `StreamEvent`.
- `on_streaming_end(payload: StreamingEndPayload, context: dict[str, Any]) -> None` *(async)* — [line 389](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L389)
  Record the `completed` span event and close the stream_with_chunking span.

## `ToolTracingPlugin`

*class* — [line 416](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L416) (`Plugin`)

Emits an `execute_tool` span per tool invocation (pre/post lifecycle).

Methods (defined on this class; inherited members not listed):

- `on_tool_pre_invoke(payload: ToolPreInvokePayload, context: dict[str, Any]) -> None` *(async)* — [line 427](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L427)
  Open the `execute_tool` span for this tool invocation.
- `on_tool_post_invoke(payload: ToolPostInvokePayload, context: dict[str, Any]) -> None` *(async)* — [line 443](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L443)
  Close the `execute_tool` span with success or error status.

## `SamplingTracingPlugin`

*class* — [line 468](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L468) (`Plugin`)

Emits a `sampling` span per sampling loop.

Methods (defined on this class; inherited members not listed):

- `on_loop_start(payload: SamplingLoopStartPayload, context: dict[str, Any]) -> None` *(async)* — [line 482](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L482)
  Open the sampling span for this loop.
- `on_iteration(payload: SamplingIterationPayload, context: dict[str, Any]) -> None` *(async)* — [line 499](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L499)
  Record a span event for one sampling attempt.
- `on_repair(payload: SamplingRepairPayload, context: dict[str, Any]) -> None` *(async)* — [line 519](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L519)
  Record a span event for one repair.
- `on_loop_end(payload: SamplingLoopEndPayload, context: dict[str, Any]) -> None` *(async)* — [line 538](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L538)
  Close the sampling span, ERROR only when the loop raised.

## `ValidationTracingPlugin`

*class* — [line 555](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L555) (`Plugin`)

Emits a `validation` span per requirement-validation batch.

Methods (defined on this class; inherited members not listed):

- `on_pre_check(payload: ValidationPreCheckPayload, context: dict[str, Any]) -> None` *(async)* — [line 566](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L566)
  Open the validation span for this check.
- `on_post_check(payload: ValidationPostCheckPayload, context: dict[str, Any]) -> None` *(async)* — [line 581](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/tracing_plugins.py#L581)
  Close the validation span, ERROR only when validation raised.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
