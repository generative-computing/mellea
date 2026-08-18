---
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

Application spans cover user-facing Mellea operations, on the
`mellea.application` tracer.

#### `session` span

Covers the lifetime of a session used as a context manager.

| Attribute | Description |
| --------- | ----------- |
| `mellea.session.id` | UUID identifying this session |
| `mellea.session.context_type` | Context class name (e.g., `SimpleContext`) |
| `gen_ai.provider.name` | Resolved provider name (e.g., `"ollama"`); set when known |

#### `start_session` span

Covers session construction (backend setup and model resolution).

| Attribute | Description |
| --------- | ----------- |
| `mellea.session.id` | UUID identifying this session |
| `mellea.session.backend_name` | Requested backend name (e.g., `"ollama"`, `"hf"`), before resolution |
| `gen_ai.request.model` | Resolved model id string |
| `mellea.session.context_type` | Context class name |

#### `action` span

One per `m.act()`, `m.instruct()`, `m.chat()`, or `@generative` call.

| Attribute | Description |
| --------- | ----------- |
| `mellea.component.type` | Component class being executed (e.g., `Instruction`) |
| `mellea.action.has_requirements` | Whether requirements were supplied |
| `mellea.action.has_strategy` | Whether a sampling strategy was supplied |
| `mellea.sampling.strategy_type` | Sampling strategy class name when present |
| `mellea.action.has_format` | Whether a format constraint was specified |
| `mellea.action.tool_calls` | Whether tool calling is enabled |
| `mellea.action.num_generate_logs` | Number of generation attempts (>1 means retries occurred) |
| `mellea.sampling.success` | Whether the sampling strategy succeeded |
| `mellea.action.response` | Model response truncated to 500 characters; recorded only when `MELLEA_TRACES_CONTENT=true` |
| `mellea.action.response_length` | Length of the model response (always recorded) |

#### `execute_tool {name}` span

One per tool the model calls, following the OTel Gen-AI tool-execution
convention.

| Attribute | Description |
| --------- | ----------- |
| `gen_ai.operation.name` | Always `"execute_tool"` |
| `gen_ai.tool.name` | Name of the invoked tool |
| `gen_ai.tool.type` | Tool type from the tool schema (e.g., `"function"`), when known |
| `gen_ai.tool.description` | Tool description from the tool schema, when known |
| `gen_ai.tool.call.id` | Provider-supplied tool-call id, when available |
| `gen_ai.tool.call.arguments` | Tool arguments truncated to 500 characters; recorded only when `MELLEA_TRACES_CONTENT=true` |
| `gen_ai.tool.call.result` | Tool result truncated to 500 characters; recorded only when `MELLEA_TRACES_CONTENT=true` |
| `mellea.tool.status` | Execution outcome (`success` or `failure`) |
| `mellea.tool.execution_time_ms` | Wall-clock tool execution time in milliseconds |
| `mellea.tool.is_control_flow` | Whether the tool is framework control flow (e.g., `final_answer` in ReAct) |
| `mellea.tool.arguments_hash` | Stable hash of the arguments; recorded independent of content capture, when the call has arguments |

#### `sampling` span

One per sampling loop, when a `m.act()`, `m.instruct()`, or `@generative` call
runs with a sampling strategy. Wraps each attempt and closes when the loop
produces a passing sample or exhausts its budget.

| Attribute | Description |
| --------- | ----------- |
| `mellea.sampling.strategy_type` | Sampling strategy class name (e.g., `RejectionSamplingStrategy`) |
| `mellea.sampling.loop_budget` | Maximum iterations per subsample |
| `mellea.sampling.requirement_count` | Number of requirements validated each iteration |
| `mellea.sampling.success` | Whether at least one attempt passed all requirements |
| `mellea.sampling.iterations_used` | Total iterations that completed across subsamples |
| `mellea.sampling.failure_reason` | Human-readable reason when `mellea.sampling.success` is `false` |

It also records an `iteration` span event per attempt and a `repair` span event
per repair.

#### `validation` span

One per requirement-validation batch, wherever requirements are checked.

| Attribute | Description |
| --------- | ----------- |
| `mellea.validation.requirement_count` | Number of requirements validated |
| `mellea.validation.passed` | Whether every requirement passed |
| `mellea.validation.passed_count` | Number of requirements that passed |
| `mellea.validation.failed_count` | Number of requirements that failed |
| `mellea.validation.failure_reasons` | List of failing requirements' reasons; recorded only when `MELLEA_TRACES_CONTENT=true` |

#### `stream_with_chunking` span

One per `stream_with_chunking()` run, wrapping the backend generation and any
per-chunk validation.

| Attribute | Description |
| --------- | ----------- |
| `mellea.streaming.has_requirements` | Whether requirements were supplied |
| `mellea.streaming.requirement_count` | Number of requirements supplied |
| `mellea.streaming.chunking_strategy` | `ChunkingStrategy` class name (e.g., `SentenceChunker`) |
| `mellea.streaming.full_text_length` | Length of the accumulated text at completion |
| `gen_ai.request.model` | Model ID, when known |
| `gen_ai.provider.name` | Provider name, when known |

It also records span events through the run: `quick_check` and `chunk` per
validated chunk, `streaming_done` once the stream drains, `full_validation`
after the final `validate()` calls, `error` on an unhandled exception, and
`completed` when the run exits.

### Backend spans (`mellea.backend`)

Backend spans cover individual LLM API calls. They follow the
[OpenTelemetry Gen-AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

| Attribute | Description |
| --------- | ----------- |
| `gen_ai.provider.name` | Backend system name mapped from class (e.g., `ollama`, `openai`) |
| `gen_ai.request.model` | Model ID requested |
| `gen_ai.operation.name` | `"chat"` for `generate_from_context`; `"text_completion"` for `generate_from_raw` |
| `gen_ai.request.stream` | `True` when streaming was requested; omitted otherwise |
| `gen_ai.output.type` | `"json"` when structured output was requested; omitted otherwise |
| `gen_ai.usage.input_tokens` | Input tokens consumed |
| `gen_ai.usage.output_tokens` | Output tokens generated |
| `gen_ai.response.model` | Actual model used in the response (may differ from request) |
| `gen_ai.response.finish_reasons` | List of finish reasons (e.g., `["stop"]`) |
| `gen_ai.response.id` | Response identifier from the backend |
| `gen_ai.response.time_to_first_chunk` | Time to first chunk in seconds; streaming requests only |

Mellea also adds context-specific attributes to backend spans:

| Attribute | Description |
| --------- | ----------- |
| `mellea.component.type` | Component type being executed |
| `mellea.request.context_size` | Number of items in context |
| `mellea.request.format_type` | Response format class name |
| `mellea.action.tool_calls` | Whether tool calling is enabled |
| `mellea.request.num_actions` | Number of actions in batch (for `generate_from_raw`) |
| `mellea.usage.total_tokens` | Total tokens reported by the backend; a Mellea extension, since semconv defines only input/output |

When `MELLEA_GENERATION_CHUNK_EVENTS=true`, backend spans also record a `chunk_processed`
span event per streamed chunk, carrying its index, added text length, and the approximate
time since the previous chunk (omitted on the first chunk). This is
opt-in and off by default, since a long response produces one event per chunk.

### Span hierarchy

Backend spans nest inside application spans:

```text
start_session             (mellea.application)
                          [gen_ai.provider.name=ollama]
                          [gen_ai.request.model=granite4.1:3b]

session                   (mellea.application)
├── action                (mellea.application)
│   │                     [mellea.component.type=Instruction]
│   └── sampling          (mellea.application)
│       │                 [mellea.sampling.strategy_type=RejectionSamplingStrategy]
│       ├── chat          (mellea.backend)
│       │                 [gen_ai.provider.name=ollama]
│       │                 [gen_ai.request.model=granite4.1:3b]
│       │                 [gen_ai.usage.input_tokens=150]
│       │                 [gen_ai.usage.output_tokens=42]
│       └── validation    (mellea.application)
│                         [mellea.validation.requirement_count=2]
├── execute_tool search   (mellea.application)
│                         [gen_ai.tool.name=search]
│                         [mellea.tool.status=success]
├── action                (mellea.application)
│   └── chat              (mellea.backend)
│                         [gen_ai.provider.name=openai]
│                         [gen_ai.request.model=gpt-4o]
└── validation            (mellea.application)   ← final m.validate() check
                          [mellea.validation.requirement_count=2]
```

Tool execution happens after the generating call completes, so `execute_tool`
spans are not nested inside the `action` that requested them. Outside a session
they are root spans.

In a `stream_with_chunking` run, the backend generation and each per-chunk
validation call nest under the `stream_with_chunking` span as sibling `chat`
spans:

```text
stream_with_chunking      (mellea.application)
│                         [mellea.streaming.chunking_strategy=SentenceChunker]
├── chat                  (mellea.backend)    ← streaming generation
│                         [gen_ai.request.model=granite4.1:3b]
└── chat                  (mellea.backend)    ← per-chunk validation
                          [gen_ai.request.model=granite4.1:3b]
```

The `stream_with_chunking` span itself parents under whatever span is active
when the run starts, or is a root span when none is.

> **Note:** Full span nesting requires Python 3.12+. On Python 3.11 some spans
> may appear flattened rather than nested; all spans and attributes are still
> emitted, only the parent-child shape differs.

## Reading traces in a typical agent run

When you open a trace in your backend, look for these patterns:

**High input token counts on early spans.** A single `action` span with
`gen_ai.usage.input_tokens` much larger than expected usually means the context
has accumulated many previous messages. Use
[prefix caching](../advanced/prefix-caching-and-kv-blocks) to reduce cost.

**Repeated `validation` spans beneath one `sampling` span.** The model is
retrying because requirements keep failing. Each `sampling` span records
`mellea.sampling.iterations_used` (how many attempts ran); open a failing attempt's
`validation` span and read `mellea.validation.failure_reasons` to see why validation is
failing (recorded as available when content capture is enabled).

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
