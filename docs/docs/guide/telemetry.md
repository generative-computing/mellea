---
title: "Telemetry"
description: "Add OpenTelemetry tracing and metrics to Mellea programs."
# diataxis: how-to
---

# Telemetry

**Prerequisites:** [Getting Started](./getting-started.md) complete,
`pip install mellea[telemetry]`, Ollama running locally.

Mellea provides built-in [OpenTelemetry](https://opentelemetry.io/) instrumentation.
Two independent trace scopes can be enabled separately, and a metrics API lets you
collect counters and histograms alongside traces. All telemetry is opt-in — if the
`[telemetry]` extra is not installed, every telemetry call is a silent no-op.

> **Note:** OpenTelemetry is an optional dependency. Mellea works normally without it.
> Install with `pip install mellea[telemetry]` or `uv pip install mellea[telemetry]`.

## Configuration

All telemetry is configured via environment variables:

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `MELLEA_TRACE_APPLICATION` | Enable application-level tracing | `false` |
| `MELLEA_TRACE_BACKEND` | Enable backend-level tracing | `false` |
| `MELLEA_TRACE_CONSOLE` | Print traces to console (debugging) | `false` |
| `MELLEA_METRICS_ENABLED` | Enable metrics collection | `false` |
| `MELLEA_METRICS_CONSOLE` | Print metrics to console (debugging) | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint for trace and metric export | none |
| `OTEL_SERVICE_NAME` | Service name in exported telemetry | `mellea` |

## Trace scopes

Mellea has two independent trace scopes:

- **`mellea.application`** — user-facing operations: session lifecycle, `@generative`
  function calls, `instruct()` and `act()` calls, sampling strategies, and requirement
  validation.
- **`mellea.backend`** — LLM backend interactions, following the
  [OpenTelemetry Gen-AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
  Records model calls, token usage, finish reasons, and API latency.

Enable both for full observability, or pick one depending on what you need to debug.

## Using `start_session()` as a context manager

Wrapping a session in `with start_session()` ties the trace lifecycle to the session
scope. All spans generated within the block are nested under the session span:

```python
from mellea import generative, start_session
from mellea.stdlib.requirements import req

@generative
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of the given text as positive, negative, or neutral."""

with start_session() as m:
    email = m.instruct(
        "Write a professional email to {{name}} about {{topic}}",
        requirements=[req("Must be formal"), req("Must be under 100 words")],
        user_variables={"name": "Alice", "topic": "project update"},
    )
    sentiment = classify_sentiment(m, text="I love this product!")
```

Run this with application tracing enabled:

```bash
export MELLEA_TRACE_APPLICATION=true
python your_script.py
```

## Debugging with console output

Print spans directly to stdout without configuring an OTLP backend:

```bash
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_CONSOLE=true
python your_script.py
```

This is the fastest way to verify that instrumentation is working.

## Exporting to an OTLP backend

Any OTLP-compatible backend works. To export to a local Jaeger instance:

```bash
# Start Jaeger
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

# Configure Mellea
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=my-mellea-app

python your_script.py
# View traces at http://localhost:16686
```

Other compatible backends include Grafana Tempo, Honeycomb, Datadog, New Relic,
AWS X-Ray (via OTLP), and Google Cloud Trace.

## Checking trace status programmatically

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

## Metrics

The metrics API exposes counters, histograms, and up-down counters backed by
the OpenTelemetry Metrics API. Enable metrics collection:

```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true   # optional: print to stdout
```

Use `create_counter` and `create_histogram` to instrument your own code:

```python
from mellea.telemetry import create_counter, create_histogram

requests = create_counter("mellea.requests", unit="1", description="Total requests")
latency = create_histogram("mellea.latency", unit="ms", description="Request latency")

requests.add(1, {"backend": "ollama", "model": "granite4:micro"})
latency.record(120, {"backend": "ollama"})
```

If `MELLEA_METRICS_ENABLED` is `false` or the `[telemetry]` extra is not installed,
all instrument calls are no-ops with no overhead.

> **Note:** Metrics are exported to `OTEL_EXPORTER_OTLP_ENDPOINT` when set.
> If metrics are enabled but no endpoint is configured and `MELLEA_METRICS_CONSOLE`
> is also `false`, Mellea will log a warning at startup.

## Span hierarchy

When both trace scopes are enabled, spans nest as follows:

```text
session_context          (mellea.application)
├── aact                 (mellea.application)
│   ├── chat             (mellea.backend) [gen_ai.system=ollama, gen_ai.request.model=granite4:micro]
│   │                    [gen_ai.usage.input_tokens=150, gen_ai.usage.output_tokens=50]
│   └── requirement_validation  (mellea.application)
└── aact                 (mellea.application)
    └── chat             (mellea.backend) [gen_ai.system=openai, gen_ai.request.model=gpt-4]
                         [gen_ai.usage.input_tokens=200, gen_ai.usage.output_tokens=75]
```

Backend spans carry Gen-AI semantic convention attributes for cross-provider comparisons:

| Attribute | Description |
| --------- | ----------- |
| `gen_ai.system` | LLM provider name (`openai`, `ollama`, `huggingface`) |
| `gen_ai.request.model` | Model requested |
| `gen_ai.response.model` | Model actually used (may differ) |
| `gen_ai.usage.input_tokens` | Input tokens consumed |
| `gen_ai.usage.output_tokens` | Output tokens generated |
| `gen_ai.response.finish_reasons` | Finish reason list (e.g., `["stop"]`) |

Application spans add Mellea-specific attributes:

| Attribute | Description |
| --------- | ----------- |
| `mellea.backend` | Backend class name |
| `mellea.action_type` | Component type being executed |
| `sampling_success` | Whether sampling succeeded |
| `num_generate_logs` | Number of generation attempts |
| `response` | Model response (truncated to 500 chars) |

> **Full example:** [`docs/examples/telemetry/telemetry_example.py`](../../examples/telemetry/telemetry_example.py)

---

**Previous:** [MCP Integration](./mcp-integration.md) |
**Next:** [Custom Sessions](./custom-sessions.md)
