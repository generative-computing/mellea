---
title: "Telemetry"
sidebarTitle: "Overview"
description: "Add OpenTelemetry tracing, metrics, and logging to Mellea programs."
# diataxis: how-to
---

**Prerequisites:** [Quick Start](../getting-started/quickstart) complete,
`pip install "mellea[telemetry]"`, Ollama running locally.

Mellea provides built-in [OpenTelemetry](https://opentelemetry.io/) instrumentation
across three independent pillars — tracing, metrics, and logging. Each can be enabled
separately. All telemetry is opt-in: if the `[telemetry]` extra is not installed,
every telemetry call is a silent no-op.

> **Note:** OpenTelemetry is an optional dependency. Mellea works normally without it.
> Install with `pip install "mellea[telemetry]"` or `uv pip install "mellea[telemetry]"`.

## Configuration

All telemetry is configured via environment variables:

### General

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `OTEL_SERVICE_NAME` | Service name for all telemetry signals | `mellea` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint for all telemetry signals | none |

### Tracing variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `MELLEA_TRACE_APPLICATION` | Enable application-level tracing | `false` |
| `MELLEA_TRACE_BACKEND` | Enable backend-level tracing | `false` |
| `MELLEA_TRACE_CONSOLE` | Print traces to console (debugging) | `false` |

### Metrics variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `MELLEA_METRICS_ENABLED` | Enable metrics collection | `false` |
| `MELLEA_METRICS_CONSOLE` | Print metrics to console (debugging) | `false` |
| `MELLEA_METRICS_OTLP` | Enable OTLP metrics exporter | `false` |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | Metrics-specific OTLP endpoint (overrides general) | none |
| `MELLEA_METRICS_PROMETHEUS` | Enable Prometheus metric reader | `false` |
| `OTEL_METRIC_EXPORT_INTERVAL` | Export interval in milliseconds | `60000` |

### Logging variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `MELLEA_LOGS_OTLP` | Enable OTLP logs exporter | `false` |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | Logs-specific OTLP endpoint (overrides general) | none |

## Quick start

Enable tracing and metrics with console output to verify everything works:

```bash
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=true
export MELLEA_TRACE_CONSOLE=true
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
python your_script.py
```

Traces and metrics print to stdout. For production use, replace the console
exporters with an OTLP endpoint:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=my-mellea-app
```

## Checking telemetry status programmatically

```python
from mellea.telemetry import (
    is_application_tracing_enabled,
    is_backend_tracing_enabled,
    is_metrics_enabled,
)

print(f"Application tracing: {is_application_tracing_enabled()}")
print(f"Backend tracing:     {is_backend_tracing_enabled()}")
print(f"Metrics:             {is_metrics_enabled()}")
```

## Tracing

Mellea has two independent trace scopes:

- **`mellea.application`** — user-facing operations: session lifecycle,
  `@generative` calls, `instruct()` and `act()`, sampling strategies, and
  requirement validation.
- **`mellea.backend`** — LLM backend interactions following the
  [OpenTelemetry Gen-AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
  Records model calls, token usage, finish reasons, and API latency.

Enable both for full observability, or pick one depending on what you need to
debug. When both scopes are active, backend spans nest inside application spans:

```text
session_context          (mellea.application)
├── aact                 (mellea.application)
│   ├── chat             (mellea.backend) [gen_ai.system=ollama]
│   └── requirement_validation  (mellea.application)
└── aact                 (mellea.application)
    └── chat             (mellea.backend) [gen_ai.system=openai]
```

See [Tracing](../evaluation-and-observability/tracing) for span attributes,
exporter configuration (Jaeger, Grafana Tempo, etc.), and debugging guidance.

## Metrics

Mellea automatically tracks token consumption across all backends using
OpenTelemetry counters (`mellea.llm.tokens.input` and
`mellea.llm.tokens.output`). No code changes are required — the
`TokenMetricsPlugin` records metrics via the plugin hook system after each
LLM call completes.

The metrics API also exposes `create_counter`, `create_histogram`, and
`create_up_down_counter` for instrumenting your own application code.

Mellea supports three exporters that can run simultaneously:

- **Console** — print to stdout for debugging
- **OTLP** — export to production observability platforms
- **Prometheus** — register with `prometheus_client` for scraping

See [Metrics](../evaluation-and-observability/metrics) for token usage details,
backend support matrix, exporter setup, custom instruments, and troubleshooting.

## Logging

Mellea uses a color-coded console logger (`FancyLogger`) by default. When the
`[telemetry]` extra is installed and `MELLEA_LOGS_OTLP=true` is set, Mellea
also exports logs to an OTLP collector alongside existing console output.

See [Logging](../evaluation-and-observability/logging) for console logging
configuration, OTLP log export setup, and programmatic access via
`get_otlp_log_handler()`.

> **Full example:** [`docs/examples/telemetry/telemetry_example.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/telemetry/telemetry_example.py)

---

**See also:**

- [Tracing](../evaluation-and-observability/tracing) — distributed traces
  with Gen-AI semantic conventions.
- [Metrics](../evaluation-and-observability/metrics) — token usage metrics,
  exporters, and custom instruments.
- [Logging](../evaluation-and-observability/logging) — console logging and OTLP
  log export.
- [Evaluate with LLM-as-a-Judge](../evaluation-and-observability/evaluate-with-llm-as-a-judge) —
  automated quality evaluation correlated with trace data.
