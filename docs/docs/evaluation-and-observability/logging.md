---
title: "Logging"
description: "Configure Mellea's console logging and export logs to OTLP collectors."
# diataxis: reference
---

**Prerequisites:** [Telemetry](../evaluation-and-observability/telemetry)
introduces the environment variables and telemetry architecture. This page
covers logging configuration in detail.

Mellea provides two logging layers: a built-in console logger for local
development and an optional OTLP exporter for centralized log aggregation.
Both work simultaneously when enabled.

## Console logging

Mellea uses `MelleaLogger`, a color-coded singleton logger built on Python's
`logging` module. All internal Mellea modules obtain their logger via
`MelleaLogger.get_logger()`.

### Configuration

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `MELLEA_LOG_LEVEL` | Log level name (e.g. `DEBUG`, `INFO`, `WARNING`) | `INFO` |
| `MELLEA_LOG_JSON` | Set to any truthy value (`1`, `true`, `yes`) to emit structured JSON instead of colour-coded output | unset |
| `MELLEA_FLOG` | Set to any value to forward logs to a local REST endpoint at `http://localhost:8000/api/receive` | unset |

By default, `MelleaLogger` logs at `INFO` level with color-coded output to
stdout. Set `MELLEA_LOG_LEVEL` to change the level:

```bash
export MELLEA_LOG_LEVEL=DEBUG
python your_script.py
```

### Log format

Console output uses ANSI color codes by log level:

- **Cyan** — DEBUG
- **Grey** — INFO
- **Yellow** — WARNING
- **Red** — ERROR
- **Bold red** — CRITICAL

Each message is formatted as:

```text
=== HH:MM:SS-LEVEL ======
message
```

## Sample output

### Console format (default)

Running `m.instruct(...)` inside a session produces lines like:

```text
=== 11:11:25-INFO ======
SUCCESS
```

### JSON format (`MELLEA_LOG_JSON=1`)

With structured JSON output enabled, the same `SUCCESS` record looks like:

```json
{
  "timestamp": "2026-04-08T11:11:25",
  "level": "INFO",
  "message": "SUCCESS",
  "module": "base",
  "function": "sample",
  "line_number": 258,
  "process_id": 73738,
  "thread_id": 6179762176,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "backend": "OllamaModelBackend",
  "model_id": "granite4:micro",
  "strategy": "RejectionSamplingStrategy",
  "loop_budget": 3
}
```

The `session_id`, `backend`, `model_id`, `strategy`, and `loop_budget` fields
are injected automatically when the call runs inside a `with session:` block.
They appear on every log record within that scope.

### Adding custom context fields

Use `log_context` to attach your own fields for the duration of a block:

```python
from mellea.core import log_context

with log_context(request_id="req-abc", user_id="usr-42"):
    result = m.instruct("Summarise this document")
    # Every log record emitted here will include request_id and user_id
```

## OTLP log export

When the `[telemetry]` extra is installed, Mellea can export logs to an OTLP
collector alongside the existing console output. This is useful for centralizing
logs from distributed services.

> **Note:** OTLP logging is disabled by default. When disabled, there is zero
> overhead — no OTLP handler is created.

### Enable OTLP logging

```bash
export MELLEA_LOGS_OTLP=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Optional: logs-specific endpoint (overrides general endpoint)
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://localhost:4318

# Optional: set service name
export OTEL_SERVICE_NAME=my-mellea-app
```

### How it works

When `MELLEA_LOGS_OTLP=true`, `MelleaLogger` adds an OpenTelemetry
`LoggingHandler` alongside its existing handlers:

- **Console handler** — continues to work normally (color-coded output)
- **REST handler** — continues to work normally (when `MELLEA_FLOG` is set)
- **OTLP handler** — exports logs to the configured OTLP collector

Logs are exported using OpenTelemetry's Logs API with batched processing
for efficiency.

### Programmatic access

Use `get_otlp_log_handler()` to add OTLP log export to your own loggers:

```python
import logging
from mellea.telemetry import get_otlp_log_handler

logger = logging.getLogger("my_app")
handler = get_otlp_log_handler()
if handler:
    logger.addHandler(handler)
    logger.info("This log will be exported via OTLP")
```

The function returns `None` when OTLP logging is disabled or not configured,
so the `if handler` check is always safe.

### OTLP collector setup example

```bash
cat > otel-collector-config.yaml <<EOF
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

exporters:
  debug:
    verbosity: detailed
  file:
    path: ./mellea-logs.json

service:
  pipelines:
    logs:
      receivers: [otlp]
      exporters: [debug, file]
EOF

docker run -p 4317:4317 \
  -v $(pwd)/otel-collector-config.yaml:/etc/otelcol/config.yaml \
  -v $(pwd):/logs \
  otel/opentelemetry-collector:latest
```

### Integration with observability platforms

OTLP logs work with any OTLP-compatible platform:

- **Grafana Loki** — log aggregation and querying
- **Elasticsearch** — log storage and analysis
- **Datadog** — unified logs, traces, and metrics
- **New Relic** — centralized logging
- **Splunk** — log analysis and monitoring

## Performance

- **Zero overhead when disabled**: No OTLP handler is created, no performance
  impact.
- **Batched export**: Logs are batched and exported asynchronously.
- **Non-blocking**: Log export never blocks application code.
- **Minimal overhead when enabled**: OpenTelemetry's efficient batching
  minimizes impact.

## Troubleshooting

**Logs not appearing in OTLP collector:**

1. Verify `MELLEA_LOGS_OTLP=true` is set.
2. Check that an OTLP endpoint is configured
   (`OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`).
3. Verify the OTLP collector is running and configured to receive logs.
4. Check collector logs for connection errors.

**Warning about missing endpoint:**

```text
WARNING: OTLP logs exporter is enabled but no endpoint is configured
```

Set either `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`.

**Connection refused:**

1. Verify the OTLP collector is running: `docker ps | grep otel`
2. Check the endpoint URL is correct (default: `http://localhost:4317`).
3. Verify network connectivity: `curl http://localhost:4317`

---

**See also:**

- [Telemetry](../evaluation-and-observability/telemetry) — overview of all
  telemetry features and configuration.
- [Tracing](../evaluation-and-observability/tracing) — distributed traces
  with Gen-AI semantic conventions.
- [Metrics](../evaluation-and-observability/metrics) — metrics, exporters,
  and custom instruments.
