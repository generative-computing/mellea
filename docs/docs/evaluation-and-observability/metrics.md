---
title: "Metrics"
description: "Automatically collect LLM metrics and instrument your own code with OpenTelemetry counters, histograms, and up-down counters."
# diataxis: how-to
---

**Prerequisites:** [Telemetry](../evaluation-and-observability/telemetry)
introduces the environment variables and telemetry architecture. This page
covers metrics collection in detail.

Mellea automatically records LLM metrics across all backends using OpenTelemetry. Metrics follow the
[Gen-AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
for standardized observability. The metrics API also lets you create your own
counters, histograms, and up-down counters for application-level instrumentation.

> **Note:** Metrics are an optional feature. All instrument calls are no-ops
> when metrics are disabled or the `[telemetry]` extra is not installed.

## Enable metrics

```bash
export MELLEA_METRICS_ENABLED=true
```

You also need at least one exporter configured — see
[Metrics export configuration](#metrics-export-configuration) below.

## Token usage metrics

Mellea records token consumption automatically after each LLM call completes.
No code changes are required.

### Token instruments

| Metric Name | Type | Unit | Description |
| ----------- | ---- | ---- | ----------- |
| `mellea.llm.tokens.input` | Counter | `tokens` | Total input/prompt tokens processed |
| `mellea.llm.tokens.output` | Counter | `tokens` | Total output/completion tokens generated |

### Token attributes

All token metrics include these attributes following Gen-AI semantic conventions:

| Attribute | Description | Example Values |
| --------- | ----------- | -------------- |
| `gen_ai.provider.name` | Backend provider name | `openai`, `ollama`, `watsonx`, `litellm`, `huggingface` |
| `gen_ai.request.model` | Model identifier | `gpt-4`, `llama3.2:7b`, `granite-3.1-8b-instruct` |

### Backend support

| Backend | Streaming | Non-Streaming | Source |
| ------- | --------- | ------------- | ------ |
| OpenAI | Yes | Yes | `usage.prompt_tokens` and `usage.completion_tokens` |
| Ollama | Yes | Yes | `prompt_eval_count` and `eval_count` |
| WatsonX | No | Yes | `input_token_count` and `generated_token_count` (streaming API limitation) |
| LiteLLM | Yes | Yes | `usage.prompt_tokens` and `usage.completion_tokens` |
| HuggingFace | Yes | Yes | Calculated from input_ids and output sequences |

> **Note:** Token usage metrics are only tracked for `generate_from_context`
> requests. `generate_from_raw` calls do not record token metrics.

### Token recording timing

Token metrics are recorded **after the full response is received**, not
incrementally during streaming:

- **Non-streaming**: Metrics recorded immediately after `await mot.avalue()` completes.
- **Streaming**: Metrics recorded after the stream is fully consumed (all chunks received).

This ensures accurate token counts from the backend's usage metadata, which
is only available after the complete response.

```python
mot, _ = await backend.generate_from_context(msg, ctx)

# Metrics NOT recorded yet (stream still in progress)
await mot.astream()

# Metrics recorded here (after stream completion)
await mot.avalue()
```

## Latency histograms

Mellea tracks request duration and time-to-first-token (TTFB) automatically
after each LLM call. No code changes are required.

### Latency instruments

| Metric Name | Type | Unit | Description |
| ----------- | ---- | ---- | ----------- |
| `mellea.llm.request.duration` | Histogram | `s` | Total request duration, from call to full response |
| `mellea.llm.ttfb` | Histogram | `s` | Time to first token (streaming requests only) |

### Latency attributes

| Attribute | Description | Example Values |
| --------- | ----------- | -------------- |
| `gen_ai.provider.name` | Backend provider name | `openai`, `ollama`, `watsonx`, `litellm`, `huggingface` |
| `gen_ai.request.model` | Model identifier | `gpt-4`, `llama3.2:7b`, `granite-3.1-8b-instruct` |
| `streaming` | Whether streaming mode was used (duration only) | `True`, `False` |

### Histogram buckets

Custom bucket boundaries are configured for LLM-sized latencies:

- **`mellea.llm.request.duration`**: `0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120` seconds
- **`mellea.llm.ttfb`**: `0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10` seconds

### Latency recording timing

- **`mellea.llm.request.duration`**: Recorded for every `generate_from_context` call,
  both streaming and non-streaming.
- **`mellea.llm.ttfb`**: Recorded only for streaming requests, measuring elapsed time
  from the `generate_from_context` call until the first chunk arrives.

Access latency data directly from a `ModelOutputThunk`:

```python
from mellea import start_session
from mellea.backends import ModelOption

with start_session() as m:
    result = m.instruct(
        "Explain quantum entanglement briefly",
        model_options={ModelOption.STREAM: True},
    )
    if result.streaming and result.ttfb_ms is not None:
        print(f"Time to first token: {result.ttfb_ms:.1f} ms")
```

## Error metrics

Mellea records LLM errors automatically after each failed backend call. No
code changes are required. Errors are classified into semantic categories for
consistent filtering across providers.

### Error counter

| Metric Name | Type | Unit | Description |
| ----------- | ---- | ---- | ----------- |
| `mellea.llm.errors` | Counter | `{error}` | Total LLM errors categorized by type |

### Error attributes

All error metrics include these attributes:

| Attribute | Description | Example Values |
| --------- | ----------- | -------------- |
| `error_type` | Semantic error category (mellea-specific) | `rate_limit`, `timeout`, `auth`, `content_policy`, `invalid_request`, `transport_error`, `server_error`, `unknown` |
| `gen_ai.provider.name` | Backend provider name | `openai`, `ollama`, `watsonx`, `litellm`, `huggingface` |
| `gen_ai.request.model` | Model identifier | `gpt-4`, `llama3.2:7b`, `granite-3.1-8b-instruct` |
| `error.type` | Python exception class name (standard OTel) | `RateLimitError`, `TimeoutError`, `AuthenticationError` |

### Error type categories

The `error_type` attribute maps exceptions to human-friendly semantic labels:

| Category | Description | Matched exceptions |
| -------- | ----------- | ------------------ |
| `rate_limit` | Request throttled by provider | `openai.RateLimitError`, class names containing `ratelimit` |
| `timeout` | Request or connection timed out | `TimeoutError`, `openai.APITimeoutError`, class names containing `timeout` |
| `auth` | Authentication or authorization failure | `openai.AuthenticationError`, `openai.PermissionDeniedError`, class names containing `auth` |
| `content_policy` | Request rejected by content moderation | `openai.BadRequestError` with `code="content_policy_violation"`, class names containing `content_policy` |
| `invalid_request` | Malformed or unsupported request | `openai.BadRequestError` (non-content-policy) |
| `transport_error` | Network or connection failure | `ConnectionError`, `openai.APIConnectionError`, class names containing `connection`/`transport` |
| `server_error` | Provider-side internal error | `openai.InternalServerError`, class names containing `server` |
| `unknown` | Unrecognized exception type | Any exception not matched above |

### When errors are recorded

Error metrics are recorded when a backend raises an exception during generation,
after the request has been dispatched to the provider. Construction-time errors
(e.g. missing API key) are not captured by the error counter.

## Cost metrics

Mellea estimates request cost automatically after each LLM call when pricing data
is available. No code changes are required.

### Cost instrument

| Metric Name | Type | Unit | Description |
| ----------- | ---- | ---- | ----------- |
| `mellea.llm.cost.usd` | Counter | `USD` | Estimated request cost in US dollars |

### Cost attributes

| Attribute | Description | Example Values |
| --------- | ----------- | -------------- |
| `gen_ai.provider.name` | Backend provider name | `openai`, `ollama`, `watsonx`, `litellm`, `huggingface` |
| `gen_ai.request.model` | Model identifier | `gpt-5.4`, `claude-sonnet-4-6` |

### Pricing data

Mellea ships with built-in pricing for current OpenAI and Anthropic models. Prices
are approximate and may become stale as providers update their rates. For models
without built-in pricing, cost is not recorded and a warning is logged instead.

### Custom pricing

Override built-in prices or add pricing for any model using a JSON file:

```bash
export MELLEA_PRICING_FILE=/path/to/my-pricing.json
```

The file format maps model IDs to per-1,000-token rates:

```json
{
  "my-custom-model": {"input_per_1k": 0.001, "output_per_1k": 0.002},
  "gpt-5.4": {"input_per_1k": 0.003, "output_per_1k": 0.015}
}
```

Custom entries override built-in prices. Errors loading the file are logged as
warnings and built-in prices are used as a fallback.

## Metrics export configuration

Mellea supports multiple metrics exporters that can be used independently or
simultaneously.

> **Warning:** If `MELLEA_METRICS_ENABLED=true` but no exporter is configured,
> Mellea logs a warning. Metrics are collected but not exported.

### Console exporter (debugging)

Print metrics to console for local debugging without setting up an
observability backend:

```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
python your_script.py
```

Metrics are printed as JSON at the configured export interval (default: 60
seconds).

### OTLP exporter (production)

Export metrics to an OTLP collector for production observability platforms
(Jaeger, Grafana, Datadog, etc.):

```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_OTLP=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Optional: metrics-specific endpoint (overrides general endpoint)
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4318

# Optional: set service name
export OTEL_SERVICE_NAME=my-mellea-app

# Optional: adjust export interval (milliseconds, default: 60000)
export OTEL_METRIC_EXPORT_INTERVAL=30000
```

**OTLP collector setup example:**

```bash
cat > otel-collector-config.yaml <<EOF
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

exporters:
  prometheus:
    endpoint: 0.0.0.0:8889
  debug:
    verbosity: detailed

service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [prometheus, debug]
EOF

docker run -p 4317:4317 -p 8889:8889 \
  -v $(pwd)/otel-collector-config.yaml:/etc/otelcol/config.yaml \
  otel/opentelemetry-collector:latest
```

### Prometheus exporter

Register metrics with the `prometheus_client` default registry for
Prometheus scraping:

```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_PROMETHEUS=true
```

When enabled, Mellea registers its OpenTelemetry metrics with the
`prometheus_client` default registry via `PrometheusMetricReader`. Your
application is responsible for exposing the registry. Common approaches:

**Standalone HTTP server** (simplest):

```python
from prometheus_client import start_http_server

start_http_server(9464)
```

**FastAPI middleware**:

```python
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

**Flask route**:

```python
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from flask import Flask, Response

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)
```

Verify with:

```bash
curl http://localhost:9464/metrics
```

**Prometheus server configuration:**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mellea'
    static_configs:
      - targets: ['localhost:9464']
```

```bash
docker run -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

Access Prometheus UI at `http://localhost:9090` and query metrics like
`mellea_llm_tokens_input`.

### Multiple exporters simultaneously

You can enable multiple exporters at once:

```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
export MELLEA_METRICS_OTLP=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export MELLEA_METRICS_PROMETHEUS=true
```

This configuration prints metrics to console for immediate feedback, exports
to an OTLP collector for centralized observability, and registers with the
`prometheus_client` registry for Prometheus scraping.

**Typical combinations:**

- **Development**: Console + Prometheus for local testing
- **Production**: OTLP + Prometheus for comprehensive monitoring
- **Debugging**: Console only for quick verification

## Custom metrics

The metrics API exposes `create_counter`, `create_histogram`, and
`create_up_down_counter` for instrumenting your own application code. These
return no-ops when metrics are disabled, so you can call them unconditionally.

```python
from mellea.telemetry import create_counter, create_histogram, create_up_down_counter

# Monotonically increasing values
requests = create_counter("myapp.requests", unit="1", description="Total requests")
requests.add(1, {"backend": "ollama", "model": "granite4:micro"})

# Value distributions
latency = create_histogram("myapp.latency", unit="ms", description="Request latency")
latency.record(120.5, {"backend": "ollama"})

# Values that increase or decrease
active = create_up_down_counter(
    "myapp.sessions.active", unit="1", description="Active sessions"
)
active.add(1)   # session started
active.add(-1)  # session ended
```

## Programmatic access

Check if metrics are enabled:

```python
from mellea.telemetry import is_metrics_enabled

if is_metrics_enabled():
    print("Metrics are being collected")
```

Access token usage and latency data from a `ModelOutputThunk`:

```python
from mellea import start_session
from mellea.backends import ModelOption

with start_session() as m:
    result = m.instruct("Write a haiku about programming")

    if result.usage:
        print(f"Prompt tokens: {result.usage['prompt_tokens']}")
        print(f"Completion tokens: {result.usage['completion_tokens']}")
        print(f"Total tokens: {result.usage['total_tokens']}")

    # Streaming mode also exposes time-to-first-token
    streamed = m.instruct(
        "Describe the solar system",
        model_options={ModelOption.STREAM: True},
    )
    print(f"Streaming: {streamed.streaming}")
    if streamed.ttfb_ms is not None:
        print(f"Time to first token: {streamed.ttfb_ms:.1f} ms")
```

The `usage` field is a dictionary with three keys: `prompt_tokens`,
`completion_tokens`, and `total_tokens`. All backends populate this field
consistently. `streaming` and `ttfb_ms` are set automatically based on whether
streaming mode was used.

## Performance

- **Zero overhead when disabled**: When `MELLEA_METRICS_ENABLED=false` (default),
  no auto-registered metrics plugins are active and all instrument calls are no-ops.
- **Minimal overhead when enabled**: Counter increments and histogram recordings
  are extremely fast (~nanoseconds per operation).
- **Async export**: Metrics are batched and exported asynchronously (default:
  every 60 seconds).
- **Non-blocking**: Metric recording never blocks LLM calls.
- **Automatic collection**: Metrics are recorded via hooks after generation
  completes — no manual instrumentation needed.

## Troubleshooting

**Metrics not appearing:**

1. Verify `MELLEA_METRICS_ENABLED=true` is set.
2. Check that at least one exporter is configured (Console, OTLP, or Prometheus).
3. For OTLP: Verify `MELLEA_METRICS_OTLP=true` and the endpoint is reachable.
4. For Prometheus: Verify `MELLEA_METRICS_PROMETHEUS=true` and your application
   exposes the registry (`curl http://localhost:PORT/metrics`).
5. Enable console output (`MELLEA_METRICS_CONSOLE=true`) to verify metrics are
   being collected.

**Missing OpenTelemetry dependency:**

```text
ImportError: No module named 'opentelemetry'
```

Install telemetry dependencies:

```bash
pip install "mellea[telemetry]"
```

**OTLP connection refused:**

```text
Failed to export metrics via OTLP
```

1. Verify the OTLP collector is running: `docker ps | grep otel`
2. Check the endpoint URL is correct (default: `http://localhost:4317`).
3. Verify network connectivity: `curl http://localhost:4317`
4. Check collector logs for errors.

**Metrics not updating:**

1. Metrics are exported at intervals (default: 60 seconds). Wait for the
   export cycle.
2. Reduce the export interval for testing:
   `export OTEL_METRIC_EXPORT_INTERVAL=10000` (10 seconds).
3. For Prometheus: Metrics update on scrape, not continuously.
4. Verify LLM calls are actually being made and completing successfully.

**No exporter configured warning:**

```text
WARNING: Metrics are enabled but no exporters are configured
```

Enable at least one exporter:

- Console: `export MELLEA_METRICS_CONSOLE=true`
- OTLP: `export MELLEA_METRICS_OTLP=true` + endpoint
- Prometheus: `export MELLEA_METRICS_PROMETHEUS=true`

> **Full example:** [`docs/examples/telemetry/metrics_example.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/telemetry/metrics_example.py)

---

**See also:**

- [Telemetry](../evaluation-and-observability/telemetry) — overview of all
  telemetry features and configuration.
- [Tracing](../evaluation-and-observability/tracing) — distributed traces
  with Gen-AI semantic conventions.
- [Logging](../evaluation-and-observability/logging) — console logging and OTLP
  log export.
