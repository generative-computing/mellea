---
canonical: "https://docs.mellea.ai/observability/tracing"
title: "Tracing"
description: "Export distributed traces from Mellea using OpenTelemetry semantic conventions."
# diataxis: how-to
---

**Prerequisites:** [Telemetry](../observability/telemetry)
introduces the environment variables and trace scopes. This page focuses on
exporting traces to external backends and interpreting the span data they contain.

Mellea instruments both user-facing operations and LLM backend calls using the
[OpenTelemetry Gen-AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
When tracing is enabled, every `m.act()`, `@generative` call, and LLM request
produces spans you can inspect in Jaeger, Grafana Tempo, Honeycomb, or any
OTLP-compatible backend.

> **Note:** Tracing is an optional feature. Mellea works normally without it.
> All telemetry calls are no-ops when the `[telemetry]` extra is not installed.

## Install and enable tracing

Install the telemetry extra:

```bash
pip install "mellea[telemetry]"
```

Enable tracing via environment variable:

```bash
export MELLEA_TRACES_ENABLED=true
```

Run your script. With tracing enabled but no exporter configured, spans are
created but discarded. To verify instrumentation immediately, add console
output:

```bash
export MELLEA_TRACES_ENABLED=true
export MELLEA_TRACES_CONSOLE=true
python your_script.py
```

Spans print to stdout in OpenTelemetry's default text format.

## Configuring an OTLP exporter

The OTLP exporter is opt-in. Enable it with `MELLEA_TRACES_OTLP=true` and set
either the trace-specific endpoint (`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) or
the general fallback (`OTEL_EXPORTER_OTLP_ENDPOINT`). Mellea uses the gRPC
OTLP exporter, so the endpoint must accept gRPC (default port 4317).

### Jaeger

```bash
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

export MELLEA_TRACES_ENABLED=true
export MELLEA_TRACES_OTLP=true
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=my-mellea-app

python your_script.py
```

Open `http://localhost:16686` to browse traces.

### Grafana Tempo

```bash
export MELLEA_TRACES_ENABLED=true
export MELLEA_TRACES_OTLP=true
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=my-mellea-app

python your_script.py
```

Grafana Tempo accepts OTLP on port 4317 by default. Point a Grafana datasource
at Tempo's HTTP endpoint (`http://localhost:3200`) and use the Explore panel to
query by service name.

### Other backends

Any OTLP-compatible backend works with the same environment variables:
Honeycomb, Datadog, New Relic, AWS X-Ray (via the OTEL collector), and
Google Cloud Trace all accept OTLP over gRPC.

### Checking trace status programmatically

```python
from mellea.telemetry import is_tracing_enabled

print(f"Tracing enabled: {is_tracing_enabled()}")
```

## What spans Mellea emits

Mellea has two trace scopes.

### Application spans (`mellea.application`)

Application spans cover user-facing Mellea operations. They appear whenever you
call `m.act()`, `m.instruct()`, `m.chat()`, or a `@generative` function.

| Attribute | Description |
| --------- | ----------- |
| `mellea.backend` | Backend class name (e.g., `OllamaModelBackend`) |
| `mellea.action_type` | Component class being executed (e.g., `Instruction`) |
| `mellea.context_size` | Length of the context at call time |
| `mellea.has_format` | Whether a format constraint was specified |
| `mellea.sampling_success` | Whether the sampling strategy succeeded |
| `mellea.num_generate_logs` | Number of generation attempts (>1 means retries occurred) |
| `mellea.response` | Model response truncated to 500 characters |

### Backend spans (`mellea.backend`)

Backend spans cover individual LLM API calls. They follow the
[OpenTelemetry Gen-AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

| Attribute | Description |
| --------- | ----------- |
| `gen_ai.provider.name` | Backend system name mapped from class (e.g., `ollama`, `openai`) |
| `gen_ai.request.model` | Model ID requested |
| `gen_ai.operation.name` | `"chat"` for `generate_from_context`; `"text_completion"` for `generate_from_raw` |
| `gen_ai.usage.input_tokens` | Input tokens consumed |
| `gen_ai.usage.output_tokens` | Output tokens generated |
| `gen_ai.usage.total_tokens` | Total tokens (input + output) |
| `gen_ai.response.model` | Actual model used in the response (may differ from request) |
| `gen_ai.response.finish_reasons` | List of finish reasons (e.g., `["stop"]`) |
| `gen_ai.response.id` | Response identifier from the backend |

Mellea also adds context-specific attributes to backend spans:

| Attribute | Description |
| --------- | ----------- |
| `mellea.backend` | Backend class name (e.g., `OpenAIBackend`) |
| `mellea.action_type` | Component type being executed |
| `mellea.context_size` | Number of items in context |
| `mellea.has_format` | Whether structured output format is specified |
| `mellea.format_type` | Response format class name |
| `mellea.tool_calls_enabled` | Whether tool calling is enabled |
| `mellea.num_actions` | Number of actions in batch (for `generate_from_raw`) |

### Span hierarchy

Backend spans nest inside application spans:

```text
session_context           (mellea.application)
├── aact                  (mellea.application)
│   │                     [mellea.action_type=Instruction]
│   │                     [mellea.backend=OllamaModelBackend]
│   ├── chat              (mellea.backend)
│   │                     [gen_ai.provider.name=ollama]
│   │                     [gen_ai.request.model=granite4.1:3b]
│   │                     [gen_ai.usage.input_tokens=150]
│   │                     [gen_ai.usage.output_tokens=42]
│   └── requirement_validation  (mellea.application)
└── aact                  (mellea.application)
    └── chat              (mellea.backend)
                          [gen_ai.provider.name=openai]
                          [gen_ai.request.model=gpt-4o]
```

## Reading traces in a typical agent run

When you open a trace in your backend, look for these patterns:

**High input token counts on early spans.** A single `aact` span with
`gen_ai.usage.input_tokens` much larger than expected usually means the context
has accumulated many previous messages. Use
[prefix caching](../advanced/prefix-caching-and-kv-blocks) to reduce cost.

**Repeated `requirement_validation` spans beneath one `aact`.** The value of
`mellea.num_generate_logs` in the parent span tells you how many retries occurred.
If the model keeps retrying, read the `mellea.response` attribute on each attempt to
understand why validation is failing.

**Long gaps between spans.** A gap between the start of a backend `chat` span
and the next application span usually indicates time spent waiting for the LLM.
This is normal for large models but worth tracking across deploys.

**`gen_ai.response.finish_reasons` containing `"length"`.** The model hit the
maximum output token limit and was cut off. Increase `max_tokens` in your
backend options or shorten your prompts.

### Full working example

The example at
[`docs/examples/telemetry/telemetry_example.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/telemetry/telemetry_example.py)
runs a session with `instruct()`, `@generative`, and `m.chat()` and prints trace
status to stdout. Run it to verify your setup:

```bash
export MELLEA_TRACES_ENABLED=true
export MELLEA_TRACES_CONSOLE=true
uv run python docs/examples/telemetry/telemetry_example.py
```

---

**See also:**

- [Telemetry](../observability/telemetry) — overview of all
  telemetry features and configuration.
- [Metrics](../observability/metrics) — metrics, exporters,
  and custom instruments.
- [Logging](../observability/logging) — console logging and OTLP
  log export.
