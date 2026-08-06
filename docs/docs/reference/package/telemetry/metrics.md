---
id: metrics
title: "mellea.telemetry.metrics"
sidebar_label: "metrics"
sidebar_position: 3
description: "OpenTelemetry metrics instrumentation for Mellea."
# diataxis: reference
---

Source: [`mellea/telemetry/metrics.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py) at commit `a535fc6345a0`.

OpenTelemetry metrics instrumentation for Mellea.

Declared exports (`__all__`): `classify_error`, `create_counter`, `create_histogram`, `create_up_down_counter`, `is_metrics_enabled`, `record_adapter_function_invocation`, `record_adapter_function_parse_failure`, `record_adapter_function_phase_duration`, `record_cost`, `record_error`, `record_request_duration`, `record_requirement_check`, `record_requirement_failure`, `record_sampling_attempt`, `record_sampling_outcome`, `record_token_usage_metrics`, `record_tool_call`, `record_ttfb`

## `is_metrics_enabled()`

*function* — [line 319](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L319)

`is_metrics_enabled() -> bool`

Check if metrics collection is enabled.

## `create_counter()`

*function* — [line 386](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L386)

`create_counter(name: str, description: str = '', unit: str = '1') -> Any`

Create a counter instrument for monotonically increasing values.

## `create_histogram()`

*function* — [line 416](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L416)

`create_histogram(name: str, description: str = '', unit: str = '1') -> Any`

Create a histogram instrument for recording value distributions.

## `create_up_down_counter()`

*function* — [line 446](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L446)

`create_up_down_counter(name: str, description: str = '', unit: str = '1') -> Any`

Create an up-down counter for values that can increase or decrease.

## `record_token_usage_metrics()`

*function* — [line 508](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L508)

`record_token_usage_metrics(input_tokens: int | None, output_tokens: int | None, model: str, provider: str) -> None`

Record token usage metrics following OpenTelemetry Gen-AI semantic conventions.

## `record_request_duration()`

*function* — [line 579](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L579)

`record_request_duration(duration_s: float, model: str, provider: str, streaming: bool = False) -> None`

Record total LLM request duration.

## `record_ttfb()`

*function* — [line 612](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L612)

`record_ttfb(ttfb_s: float, model: str, provider: str) -> None`

Record time-to-first-token for streaming LLM requests.

## `classify_error()`

*function* — [line 653](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L653)

`classify_error(exc: BaseException) -> str`

Map an exception to a semantic error type string.

## `record_error()`

*function* — [line 736](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L736)

`record_error(error_type: str, model: str, provider: str, exception_class: str) -> None`

Record an LLM error metric.

## `record_cost()`

*function* — [line 797](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L797)

`record_cost(cost: float, model: str, provider: str) -> None`

Record estimated LLM request cost in USD.

## `record_sampling_attempt()`

*function* — [line 867](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L867)

`record_sampling_attempt(strategy: str) -> None`

Record one sampling attempt for the given strategy.

## `record_sampling_outcome()`

*function* — [line 881](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L881)

`record_sampling_outcome(strategy: str, success: bool) -> None`

Record the final outcome (success or failure) of a sampling loop.

## `record_requirement_check()`

*function* — [line 929](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L929)

`record_requirement_check(requirement: str) -> None`

Record one requirement validation check.

## `record_requirement_failure()`

*function* — [line 943](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L943)

`record_requirement_failure(requirement: str, reason: str) -> None`

Record one requirement validation failure.

## `record_tool_call()`

*function* — [line 976](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L976)

`record_tool_call(tool: str, status: str) -> None`

Record one tool invocation.

## `record_adapter_function_invocation()`

*function* — [line 1043](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L1043)

`record_adapter_function_invocation(name: str, revision: str | None, binding_type: str, adapter_type: str, outcome: str) -> None`

Record one adapter function invocation.

## `record_adapter_function_phase_duration()`

*function* — [line 1074](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L1074)

`record_adapter_function_phase_duration(name: str, phase: str, duration_s: float) -> None`

Record the duration of one adapter function lifecycle phase.

## `record_adapter_function_parse_failure()`

*function* — [line 1096](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/metrics.py#L1096)

`record_adapter_function_parse_failure(name: str, revision: str | None) -> None`

Record one adapter function schema-mismatch parse failure.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
