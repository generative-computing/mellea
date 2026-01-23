# OpenTelemetry Tracing Examples

This directory contains examples demonstrating OpenTelemetry instrumentation in Mellea.

## Quick Start

### 1. Install Dependencies

```bash
uv sync --all-extras
```

### 2. Start Ollama (Required)

```bash
ollama serve
```

### 3. Run Examples

#### Basic Example (No Tracing)

```bash
python telemetry_example.py
```

#### With Application Tracing Only

```bash
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=false
python telemetry_example.py
```

#### With Backend Tracing Only

```bash
export MELLEA_TRACE_APPLICATION=false
export MELLEA_TRACE_BACKEND=true
python telemetry_example.py
```

#### With Both Traces

```bash
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=true
python telemetry_example.py
```

#### With Console Output (Debugging)

```bash
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=true
export MELLEA_TRACE_CONSOLE=true
python telemetry_example.py
```

## Using Jaeger for Visualization

### 1. Start Jaeger

```bash
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

### 2. Configure Mellea

```bash
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=mellea-example
```

### 3. Run Example

```bash
python telemetry_example.py
```

### 4. View Traces

Open http://localhost:16686 in your browser and select "mellea-example" service.

## What Gets Traced

### Application Trace (`mellea.application`)

- Session lifecycle (start, enter, exit)
- @generative function calls
- Action execution (aact)
- Sampling strategies
- Requirement validation

### Backend Trace (`mellea.backend`)

- Model generation calls
- Context-based generation
- Raw generation
- Backend-specific operations (Ollama, OpenAI, etc.)

## Trace Attributes

Each span includes rich metadata:

- **model_id**: Model identifier
- **backend**: Backend class name
- **action_type**: Component type
- **context_size**: Number of context items
- **has_requirements**: Whether requirements are specified
- **strategy_type**: Sampling strategy used
- **tool_calls**: Whether tool calling is enabled
- **format_type**: Response format class

## Performance Impact

- **Disabled (default)**: Near-zero overhead
- **Application only**: Minimal overhead (~1-2%)
- **Backend only**: Minimal overhead (~1-2%)
- **Both enabled**: Low overhead (~2-5%)

## Troubleshooting

**Traces not appearing in Jaeger:**
1. Check Jaeger is running: `docker ps | grep jaeger`
2. Verify endpoint: `curl http://localhost:4317`
3. Check environment variables are set
4. Enable console output to verify traces are created

**Import errors:**
```bash
uv sync  # Reinstall dependencies
```

**Ollama connection errors:**
```bash
ollama serve  # Start Ollama server
```

## Learn More

See [docs/dev/telemetry.md](../../dev/telemetry.md) for complete documentation.