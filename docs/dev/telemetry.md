## OpenTelemetry Instrumentation in Mellea

Mellea provides built-in OpenTelemetry instrumentation with comprehensive observability features that can be enabled independently. The instrumentation follows the [OpenTelemetry Gen-AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) for standardized observability across LLM applications.

**Note**: OpenTelemetry is an optional dependency. If not installed, telemetry features are automatically disabled with no impact on functionality.

### Observability Features

1. **Application Trace** (`mellea.application`) - Tracks user-facing operations
2. **Backend Trace** (`mellea.backend`) - Tracks LLM backend interactions with Gen-AI semantic conventions
3. **Token Usage Metrics** - Tracks token consumption across all backends with Gen-AI semantic conventions

### Installation

To use telemetry features, install Mellea with OpenTelemetry support:

```bash
pip install mellea[telemetry]
# or
uv pip install mellea[telemetry]
```

Without the `[telemetry]` extra, Mellea works normally but telemetry features are disabled.

### Configuration

Telemetry is configured via environment variables:

#### General Telemetry Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_SERVICE_NAME` | Service name for all telemetry signals | `mellea` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint for all telemetry signals | None |

#### Tracing Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MELLEA_TRACE_APPLICATION` | Enable application-level tracing | `false` |
| `MELLEA_TRACE_BACKEND` | Enable backend-level tracing | `false` |
| `MELLEA_TRACE_CONSOLE` | Print traces to console (debugging) | `false` |

#### Metrics Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MELLEA_METRICS_ENABLED` | Enable metrics collection | `false` |
| `MELLEA_METRICS_CONSOLE` | Print metrics to console (debugging) | `false` |
| `MELLEA_METRICS_OTLP` | Enable OTLP metrics exporter | `false` |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | OTLP metrics-specific endpoint (overrides general endpoint) | None |
| `MELLEA_METRICS_PROMETHEUS` | Enable Prometheus metric reader (registers with prometheus_client registry) | `false` |
| `OTEL_METRIC_EXPORT_INTERVAL` | Export interval in milliseconds | `60000` |

### Application Trace Scope

The application tracer (`mellea.application`) instruments:

- **Session lifecycle**: `start_session()`, session context manager entry/exit
- **@generative functions**: Execution of functions decorated with `@generative`
- **mfuncs.aact()**: Action execution with requirements and sampling strategies
- **Sampling strategies**: Rejection sampling, budget forcing, etc.
- **Requirement validation**: Validation of requirements and constraints

**Span attributes include:**
- `backend`: Backend class name
- `model_id`: Model identifier
- `context_type`: Context class name
- `action_type`: Component type being executed
- `has_requirements`: Whether requirements are specified
- `has_strategy`: Whether a sampling strategy is used
- `strategy_type`: Sampling strategy class name
- `num_generate_logs`: Number of generation attempts
- `sampling_success`: Whether sampling succeeded
- `response`: Model response (truncated to 500 chars)
- `response_length`: Full length of model response

### Backend Trace Scope

The backend tracer (`mellea.backend`) instruments LLM interactions following [OpenTelemetry Gen-AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

- **Backend.generate_from_context()**: Context-based generation (chat operations)
- **Backend.generate_from_raw()**: Raw generation without context (text completions)
- **Backend-specific implementations**: Ollama, OpenAI, HuggingFace, Watsonx, LiteLLM

**Gen-AI Semantic Convention Attributes:**
- `gen_ai.system`: LLM system name (e.g., `openai`, `ollama`, `huggingface`)
- `gen_ai.request.model`: Model identifier used for the request
- `gen_ai.response.model`: Actual model used in the response (may differ from request)
- `gen_ai.operation.name`: Operation type (`chat` or `text_completion`)
- `gen_ai.usage.input_tokens`: Number of input tokens consumed
- `gen_ai.usage.output_tokens`: Number of output tokens generated
- `gen_ai.usage.total_tokens`: Total tokens consumed
- `gen_ai.response.id`: Response ID from the LLM provider
- `gen_ai.response.finish_reasons`: List of finish reasons (e.g., `["stop"]`, `["length"]`)

**Mellea-Specific Attributes:**
- `mellea.backend`: Backend class name (e.g., `OpenAIBackend`)
- `mellea.action_type`: Component type being executed
- `mellea.context_size`: Number of items in context
- `mellea.has_format`: Whether structured output format is specified
- `mellea.format_type`: Response format class name
- `mellea.tool_calls_enabled`: Whether tool calling is enabled
- `mellea.num_actions`: Number of actions in batch (for `generate_from_raw`)

### Token Usage Metrics

Mellea automatically tracks token consumption across backends using OpenTelemetry metrics counters. Token metrics follow the [Gen-AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) for standardized observability.

> **Note**: Token usage metrics are only tracked for `generate_from_context` requests. `generate_from_raw` calls do not record token metrics.

#### Metrics

| Metric Name | Type | Unit | Description |
|-------------|------|------|-------------|
| `mellea.llm.tokens.input` | Counter | `tokens` | Total input/prompt tokens processed |
| `mellea.llm.tokens.output` | Counter | `tokens` | Total output/completion tokens generated |

#### Attributes

All token metrics include these attributes following Gen-AI semantic conventions:

| Attribute | Description | Example Values |
|-----------|-------------|----------------|
| `gen_ai.system` | Backend system name | `openai`, `ollama`, `watsonx`, `litellm`, `huggingface` |
| `gen_ai.request.model` | Model identifier | `gpt-4`, `llama3.2:7b`, `granite-3.1-8b-instruct` |
| `mellea.backend` | Backend class name | `OpenAIBackend`, `OllamaBackend`, `WatsonxBackend` |

#### Backend Support

| Backend | Streaming | Non-Streaming | Source |
|---------|-----------|---------------|--------|
| OpenAI | ✅ | ✅ | `usage.prompt_tokens` and `usage.completion_tokens` |
| Ollama | ✅ | ✅ | `prompt_eval_count` and `eval_count` |
| WatsonX | ❌ | ✅ | `input_token_count` and `generated_token_count` (streaming API limitation) |
| LiteLLM | ✅ | ✅ | `usage.prompt_tokens` and `usage.completion_tokens` |
| HuggingFace | ✅ | ✅ | Calculated from input_ids and output sequences |

#### Configuration

Token metrics are **disabled by default** for zero overhead. Enable with:

```bash
export MELLEA_METRICS_ENABLED=true
```

Metrics are automatically recorded after each LLM call completes. No code changes required.

#### When Metrics Are Recorded

Token metrics are recorded **after the full response is received**, not incrementally during streaming:

- **Non-streaming**: Metrics recorded immediately after `await mot.avalue()` completes
- **Streaming**: Metrics recorded after the stream is fully consumed (all chunks received)

This ensures accurate token counts are captured from the backend's usage metadata, which is only available after the complete response.

```python
mot, _ = await backend.generate_from_context(msg, ctx)

# Metrics NOT recorded yet (stream still in progress)
await mot.astream()

# Metrics recorded here (after stream completion)
await mot.avalue()
```

#### Metrics Export Configuration

Mellea supports multiple metrics exporters that can be used independently or simultaneously:

1. **Console Exporter** - For local debugging
2. **OTLP Exporter** - For production observability platforms
3. **Prometheus Exporter** - For Prometheus-based monitoring

**Important**: If `MELLEA_METRICS_ENABLED=true` but no exporter is configured, you'll see a warning. Metrics will be collected but not exported.

##### Console Exporter (Debugging)

Print metrics to console for local debugging without setting up an observability backend:

```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
python docs/examples/telemetry/metrics_example.py
```

Metrics are printed as JSON at the configured export interval (default: 60 seconds).

##### OTLP Exporter (Production)

Export metrics to an OTLP collector for production observability platforms (Jaeger, Grafana, Datadog, etc.):

```bash
# Enable metrics and OTLP exporter
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_OTLP=true

# Configure OTLP endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Optional: Use metrics-specific endpoint (overrides general endpoint)
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4318

# Optional: Set service name
export OTEL_SERVICE_NAME=my-mellea-app

# Optional: Adjust export interval (milliseconds, default: 60000)
export OTEL_METRIC_EXPORT_INTERVAL=30000

python docs/examples/telemetry/metrics_example.py
```

**OTLP Collector Setup Example:**

```bash
# Create otel-collector-config.yaml
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

# Start OTLP collector
docker run -p 4317:4317 -p 8889:8889 \
  -v $(pwd)/otel-collector-config.yaml:/etc/otelcol/config.yaml \
  otel/opentelemetry-collector:latest
```

##### Prometheus Exporter

Register metrics with the `prometheus_client` default registry for Prometheus scraping:

```bash
# Enable metrics and Prometheus metric reader
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_PROMETHEUS=true

# Optional: Set service name
export OTEL_SERVICE_NAME=my-mellea-app

python docs/examples/telemetry/metrics_example.py
```

When enabled, Mellea registers its OpenTelemetry metrics with the `prometheus_client` default registry via `PrometheusMetricReader`. Your application is responsible for exposing the registry for scraping. Common approaches:

**Standalone HTTP server** (simplest):

```python
from prometheus_client import start_http_server

# Start Prometheus scrape endpoint on port 9464
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

**Verify Prometheus Endpoint:**

```bash
curl http://localhost:9464/metrics
```

**Prometheus Server Configuration:**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mellea'
    static_configs:
      - targets: ['localhost:9464']
```

**Start Prometheus:**

```bash
docker run -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

Access Prometheus UI at `http://localhost:9090` and query metrics like `mellea_llm_tokens_input`.

##### Multiple Exporters Simultaneously

You can enable multiple exporters at once for different use cases:

```bash
# Enable all exporters
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true          # Debug output
export MELLEA_METRICS_OTLP=true             # Production observability
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export MELLEA_METRICS_PROMETHEUS=true       # Prometheus monitoring

python docs/examples/telemetry/metrics_example.py
```

This configuration:
- Prints metrics to console for immediate feedback
- Exports to OTLP collector for centralized observability
- Registers metrics with prometheus_client registry for Prometheus scraping

**Use Cases:**
- **Development**: Console + Prometheus for local testing
- **Production**: OTLP + Prometheus for comprehensive monitoring
- **Debugging**: Console only for quick verification

#### Programmatic Access

Check if metrics are enabled:

```python
from mellea.telemetry import is_metrics_enabled

if is_metrics_enabled():
    print("Token metrics are being collected")
```

#### Performance

- **Zero overhead when disabled**: When `MELLEA_METRICS_ENABLED=false` (default), `record_token_usage_metrics()` returns immediately with no processing
- **Minimal overhead when enabled**: Counter increments are extremely fast (~nanoseconds per operation)
- **Async export**: Metrics are batched and exported asynchronously (default: every 60 seconds)
- **Non-blocking**: Metric recording never blocks LLM calls

#### Use Cases

**Cost Monitoring**: Track token consumption to estimate and control LLM costs across models and backends.

**Performance Optimization**: Identify operations consuming excessive tokens and optimize prompts.

**Model Comparison**: Compare token efficiency across different models for the same tasks.

**Budget Enforcement**: Set up alerts when token usage exceeds thresholds.

**Capacity Planning**: Analyze token usage patterns to plan infrastructure capacity.

### Trace Usage Examples

#### Enable Application Tracing Only

```bash
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=false
python docs/examples/instruct_validate_repair/101_email.py
```

This traces user-facing operations like `@generative` function calls, session lifecycle, and sampling strategies, but not the underlying LLM API calls.

#### Enable Backend Tracing Only

```bash
export MELLEA_TRACE_APPLICATION=false
export MELLEA_TRACE_BACKEND=true
python docs/examples/instruct_validate_repair/101_email.py
```

This traces only the LLM backend interactions, showing model calls, token usage, and API latency.

#### Enable Both Traces

```bash
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=true
python docs/examples/instruct_validate_repair/101_email.py
```

This provides complete observability across both application logic and backend interactions.

#### Export to Jaeger

```bash
# Start Jaeger (example using Docker)
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

# Configure Mellea to export traces
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=my-mellea-app

python docs/examples/instruct_validate_repair/101_email.py

# View traces at http://localhost:16686
```

#### Console Output for Debugging

```bash
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_CONSOLE=true
python docs/examples/instruct_validate_repair/101_email.py
```

This prints trace spans to the console, useful for local debugging without setting up a trace backend.

### Programmatic Access

You can check if tracing is enabled in your code:

```python
from mellea.telemetry import (
    is_application_tracing_enabled,
    is_backend_tracing_enabled,
)

if is_application_tracing_enabled():
    print("Application tracing is enabled")

if is_backend_tracing_enabled():
    print("Backend tracing is enabled")
```

### Performance Considerations

- **Zero overhead when disabled**: When tracing is disabled (default), there is minimal performance impact
- **Async-friendly**: Tracing works seamlessly with async operations
- **Batched export**: Traces are exported in batches to minimize network overhead
- **Separate scopes**: Enable only the tracing you need to reduce overhead

### Integration with Observability Tools

Mellea's OpenTelemetry instrumentation works with any OTLP-compatible backend:

- **Jaeger**: Distributed tracing
- **Zipkin**: Distributed tracing
- **Grafana Tempo**: Distributed tracing
- **Honeycomb**: Observability platform
- **Datadog**: APM and observability
- **New Relic**: APM and observability
- **AWS X-Ray**: Distributed tracing (via OTLP)
- **Google Cloud Trace**: Distributed tracing (via OTLP)

### Example Trace Hierarchy

When both traces are enabled, you'll see a hierarchy like:

```
session_context (application)
├── aact (application)
│   ├── chat (backend) [gen_ai.system=ollama, gen_ai.request.model=llama3.2]
│   │   └── [gen_ai.usage.input_tokens=150, gen_ai.usage.output_tokens=50]
│   └── requirement_validation (application)
├── aact (application)
│   └── chat (backend) [gen_ai.system=openai, gen_ai.request.model=gpt-4]
│       └── [gen_ai.usage.input_tokens=200, gen_ai.usage.output_tokens=75]
```

The Gen-AI semantic conventions make it easy to:
- Track token usage across different LLM providers
- Compare performance between models
- Monitor costs based on token consumption
- Identify which operations consume the most tokens

### Troubleshooting

#### Traces

**Traces not appearing:**
1. Verify environment variables are set correctly
2. Check that OTLP endpoint is reachable
3. Enable console output to verify traces are being created
4. Check firewall/network settings

**High overhead:**
1. Disable application tracing if you only need backend metrics
2. Reduce sampling rate (future feature)
3. Use a local OTLP collector to batch exports

**Missing spans:**
1. Ensure you're using `with start_session()` context manager
2. Check that async operations are properly awaited
3. Verify backend implementation has instrumentation

#### Metrics

**Metrics not appearing:**
1. Verify `MELLEA_METRICS_ENABLED=true` is set
2. Check that at least one exporter is configured (Console, OTLP, or Prometheus)
3. For OTLP: Verify `MELLEA_METRICS_OTLP=true` and endpoint is reachable
4. For Prometheus: Verify `MELLEA_METRICS_PROMETHEUS=true` and your application exposes the registry (`curl http://localhost:PORT/metrics`)
5. Enable console output (`MELLEA_METRICS_CONSOLE=true`) to verify metrics are being collected

**Missing OpenTelemetry dependency:**
```
ImportError: No module named 'opentelemetry'
```
Install telemetry dependencies:
```bash
pip install mellea[telemetry]
# or
uv pip install mellea[telemetry]
```

**OTLP connection refused:**
```
Failed to export metrics via OTLP
```
1. Verify OTLP collector is running: `docker ps | grep otel`
2. Check endpoint URL is correct (default: `http://localhost:4317`)
3. Verify network connectivity: `curl http://localhost:4317`
4. Check collector logs for errors

**Metrics not updating:**
1. Metrics are exported at intervals (default: 60 seconds). Wait for export cycle.
2. Reduce export interval for testing: `export OTEL_METRIC_EXPORT_INTERVAL=10000` (10 seconds)
3. For Prometheus: Metrics update on scrape, not continuously
4. Verify LLM calls are actually being made and completing successfully

**No exporter configured warning:**
```
WARNING: Metrics are enabled but no exporters are configured
```
Enable at least one exporter:
- Console: `export MELLEA_METRICS_CONSOLE=true`
- OTLP: `export MELLEA_METRICS_OTLP=true` + endpoint
- Prometheus: `export MELLEA_METRICS_PROMETHEUS=true`

### Future Enhancements

Planned improvements to telemetry:

**Metrics:**
- Latency histograms: Request duration and time-to-first-token (TTFB) for streaming
- Cost tracking: Automatic cost estimation based on token usage and model pricing
- Error counters: Categorized by semantic error type (rate limits, timeouts, content policy, etc.)
- Operational metrics: Sampling attempts, requirement validation outcomes, tool calls

**Tracing:**
- Sampling rate configuration
- Custom span attributes via decorators
- Trace context propagation for distributed systems
- Integration with LangSmith and other LLM observability tools

**Logging:**
- Structured logging with OpenTelemetry Logs API
- Log correlation with traces and metrics
- Log export to OTLP collectors