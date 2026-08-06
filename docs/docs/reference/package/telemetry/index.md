---
id: index
title: "mellea.telemetry"
sidebar_label: "Overview"
sidebar_position: 0
description: "OpenTelemetry instrumentation for Mellea."
# diataxis: reference
---

Source: [`mellea/telemetry/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/telemetry/__init__.py) at commit `a535fc6345a0`.

OpenTelemetry instrumentation for Mellea.

Declared exports (`__all__`): `MelleaContextFilter`, `async_with_context`, `create_counter`, `create_histogram`, `create_up_down_counter`, `generate_request_id`, `get_current_context`, `get_model_id`, `get_otlp_log_handler`, `get_request_id`, `get_sampling_iteration`, `get_session_id`, `is_content_tracing_enabled`, `is_metrics_enabled`, `is_pricing_enabled`, `is_tracing_enabled`, `record_adapter_function_invocation`, `record_adapter_function_parse_failure`, `record_adapter_function_phase_duration`, `record_cost`, `record_error`, `record_request_duration`, `record_requirement_check`, `record_requirement_failure`, `record_sampling_attempt`, `record_sampling_outcome`, `record_token_usage_metrics`, `record_tool_call`, `record_ttfb`, `with_context`

## Modules

- [`mellea.telemetry.context`](context.md) — Async-safe context propagation for Mellea telemetry.
- [`mellea.telemetry.logging`](logging.md) — OpenTelemetry logging instrumentation for Mellea.
- [`mellea.telemetry.metrics`](metrics.md) — OpenTelemetry metrics instrumentation for Mellea.
- [`mellea.telemetry.metrics_plugins`](metrics_plugins.md) — Metrics plugins for recording telemetry data via hooks.
- [`mellea.telemetry.pricing`](pricing.md) — LLM pricing via litellm's pricing API.
- [`mellea.telemetry.tracing`](tracing.md) — OpenTelemetry tracing instrumentation for Mellea.
- [`mellea.telemetry.tracing_plugins`](tracing_plugins.md) — Tracing plugins for emitting OpenTelemetry spans via hooks.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
