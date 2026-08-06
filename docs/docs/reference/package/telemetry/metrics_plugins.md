---
id: metrics_plugins
title: "mellea.telemetry.metrics_plugins"
sidebar_label: "metrics_plugins"
sidebar_position: 4
description: "Metrics plugins for recording telemetry data via hooks."
# diataxis: reference
---

Source: [`mellea/telemetry/metrics_plugins.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py) at commit `a535fc6345a0`.

Metrics plugins for recording telemetry data via hooks.

## `TokenMetricsPlugin`

*class* — [line 52](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L52) (`Plugin`)

Records token usage metrics from generation outputs.

Methods (defined on this class; inherited members not listed):

- `record_token_metrics(payload: GenerationPostCallPayload, context: dict[str, Any]) -> None` *(async)* — [line 64](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L64)
  Record token metrics after generation completes.
- `record_batch_token_metrics(payload: GenerationBatchPostCallPayload, context: dict[str, Any]) -> None` *(async)* — [line 88](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L88)
  Record token metrics after a batch generation completes.

## `LatencyMetricsPlugin`

*class* — [line 110](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L110) (`Plugin`)

Records request duration and TTFB latency metrics from generation outputs.

Methods (defined on this class; inherited members not listed):

- `record_latency_metrics(payload: GenerationPostCallPayload, context: dict[str, Any]) -> None` *(async)* — [line 120](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L120)
  Record latency metrics after generation completes.
- `record_batch_latency_metrics(payload: GenerationBatchPostCallPayload, context: dict[str, Any]) -> None` *(async)* — [line 148](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L148)
  Record request duration after a batch generation completes.

## `ErrorMetricsPlugin`

*class* — [line 170](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L170) (`Plugin`)

Records LLM error counts from generation errors.

Methods (defined on this class; inherited members not listed):

- `record_error_metrics(payload: GenerationErrorPayload, context: dict[str, Any]) -> None` *(async)* — [line 179](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L179)
  Record error metrics when a generation error occurs.
- `record_batch_error_metrics(payload: GenerationBatchErrorPayload, context: dict[str, Any]) -> None` *(async)* — [line 205](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L205)
  Record error metrics when a batch generation fails.
- `record_streaming_error_metrics(payload: StreamingEndPayload, context: dict[str, Any]) -> None` *(async)* — [line 225](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L225)
  Record error metrics when `stream_with_chunking` ends with an exception.

## `CostMetricsPlugin`

*class* — [line 247](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L247) (`Plugin`)

Records estimated request cost metrics from generation outputs.

Methods (defined on this class; inherited members not listed):

- `record_cost_metrics(payload: GenerationPostCallPayload, context: dict[str, Any]) -> None` *(async)* — [line 257](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L257)
  Record cost metrics after generation completes.
- `record_batch_cost_metrics(payload: GenerationBatchPostCallPayload, context: dict[str, Any]) -> None` *(async)* — [line 293](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L293)
  Record cost metrics after a batch generation completes.

## `SamplingMetricsPlugin`

*class* — [line 328](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L328) (`Plugin`)

Records sampling loop attempt and outcome metrics.

Methods (defined on this class; inherited members not listed):

- `record_sampling_attempt(payload: SamplingIterationPayload, context: dict[str, Any]) -> None` *(async)* — [line 336](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L336)
  Record one sampling attempt after each iteration.
- `record_sampling_outcome(payload: SamplingLoopEndPayload, context: dict[str, Any]) -> None` *(async)* — [line 350](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L350)
  Record success or failure when the sampling loop ends, unless it raised.
- `record_streaming_outcome(payload: StreamingEndPayload, context: dict[str, Any]) -> None` *(async)* — [line 368](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L368)
  Record the `stream_with_chunking` outcome when the orchestrator finishes.

## `RequirementMetricsPlugin`

*class* — [line 382](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L382) (`Plugin`)

Records requirement validation check and failure metrics.

Methods (defined on this class; inherited members not listed):

- `record_requirement_metrics(payload: ValidationPostCheckPayload, context: dict[str, Any]) -> None` *(async)* — [line 390](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L390)
  Record validation checks and failures for each requirement, unless it raised.
- `record_streaming_requirement_metrics(payload: StreamingEventPayload, context: dict[str, Any]) -> None` *(async)* — [line 420](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L420)
  Record per-chunk requirement metrics for `QuickCheckEvent`s.

## `ToolMetricsPlugin`

*class* — [line 446](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L446) (`Plugin`)

Records tool invocation metrics.

Methods (defined on this class; inherited members not listed):

- `record_tool_call(payload: ToolPostInvokePayload, context: dict[str, Any]) -> None` *(async)* — [line 453](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L453)
  Record one tool invocation after it completes.

## `AdapterFunctionMetricsPlugin`

*class* — [line 473](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L473) (`Plugin`)

Records adapter function invocation and phase-duration metrics.

Methods (defined on this class; inherited members not listed):

- `record_adapter_function_invocation(payload: AdapterFunctionInvocationCompletePayload, context: dict[str, Any]) -> None` *(async)* — [line 486](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L486)
  Record one adapter function invocation after it completes.
- `record_adapter_function_phase(payload: AdapterFunctionPhaseCompletePayload, context: dict[str, Any]) -> None` *(async)* — [line 511](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics_plugins.py#L511)
  Record one adapter function lifecycle phase after it completes.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
