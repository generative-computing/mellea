## OpenTelemetry Instrumentation in Mellea

Mellea provides built-in OpenTelemetry instrumentation with two independent trace scopes that can be enabled separately:

1. **Application Trace** (`mellea.application`) - Tracks user-facing operations
2. **Backend Trace** (`mellea.backend`) - Tracks LLM backend interactions

### Configuration

Telemetry is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MELLEA_TRACE_APPLICATION` | Enable application-level tracing | `false` |
| `MELLEA_TRACE_BACKEND` | Enable backend-level tracing | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint for trace export | None |
| `OTEL_SERVICE_NAME` | Service name for traces | `mellea` |
| `MELLEA_TRACE_CONSOLE` | Print traces to console (debugging) | `false` |

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

The backend tracer (`mellea.backend`) instruments:

- **Backend.generate_from_context()**: Context-based generation
- **Backend.generate_from_raw()**: Raw generation without context
- **Backend-specific implementations**: Ollama, OpenAI, HuggingFace, Watsonx, LiteLLM

**Span attributes include:**
- `backend`: Backend class name (e.g., `OllamaModelBackend`)
- `model_id`: Model identifier string
- `action_type`: Component type
- `context_size`: Number of items in context
- `has_format`: Whether structured output format is specified
- `format_type`: Response format class name
- `tool_calls`: Whether tool calling is enabled
- `num_actions`: Number of actions in batch (for `generate_from_raw`)

### Usage Examples

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
│   ├── generate_from_context (backend)
│   │   └── ollama.chat (backend)
│   └── requirement_validation (application)
├── aact (application)
│   └── generate_from_context (backend)
│       └── ollama.chat (backend)
```

### Troubleshooting

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

### Future Enhancements

Planned improvements to telemetry:

- Sampling rate configuration
- Custom span attributes via decorators
- Metrics export (token counts, latency percentiles)
- Trace context propagation for distributed systems
- Integration with LangSmith and other LLM observability tools