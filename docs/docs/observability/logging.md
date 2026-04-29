---
title: "Logging"
description: "Configure Mellea's console logging and export logs to OTLP collectors."
# diataxis: how-to
---

**Prerequisites:** [Telemetry](../observability/telemetry)
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
| `MELLEA_LOG_ENABLED` | Master switch. Set to `false` / `0` / `no` to suppress all handlers (useful in test environments) | `true` |
| `MELLEA_LOG_LEVEL` | Log level name (e.g. `DEBUG`, `INFO`, `WARNING`) | `INFO` |
| `MELLEA_LOG_JSON` | Set to any truthy value (`1`, `true`, `yes`) to emit structured JSON instead of colour-coded output | unset |
| `MELLEA_LOG_CONSOLE` | Set to `false` / `0` / `no` to disable the console (stdout) handler | `true` |
| `MELLEA_LOG_FILE` | Absolute or relative path for rotating file output (e.g. `/var/log/mellea.log`). When unset no file handler is attached | unset |
| `MELLEA_LOG_FILE_MAX_BYTES` | Maximum size in bytes before the log file is rotated | `10485760` (10 MB) |
| `MELLEA_LOG_FILE_BACKUP_COUNT` | Number of rotated backup files to keep | `5` |
| `MELLEA_LOG_OTLP` | Set to `true` / `1` / `yes` to export logs via OTLP. Requires `opentelemetry-sdk` and a configured OTLP endpoint | `false` |
| `MELLEA_LOG_WEBHOOK` | HTTP(S) URL to forward log records to via HTTP POST. Supersedes the deprecated `MELLEA_FLOG` and `FLOG` variables | unset |
| `MELLEA_FLOG` | **Deprecated.** Activates a webhook handler pointed at `http://localhost:8000/api/receive`. Use `MELLEA_LOG_WEBHOOK` instead | unset |

> **Note:** If `MELLEA_LOG_FILE` is set but the path cannot be opened (for
> example due to a permissions error or an invalid path), Mellea emits a
> `UserWarning` and continues without file logging. The application is never
> crashed by a misconfigured log path.

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
  "model_id": "granite4.1:3b",
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
export MELLEA_LOG_OTLP=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Optional: log-specific endpoint (overrides general endpoint)
export OTEL_EXPORTER_OTLP_LOG_ENDPOINT=http://localhost:4318

# Optional: set service name
export OTEL_SERVICE_NAME=my-mellea-app
```

### How it works

When `MELLEA_LOG_OTLP=true`, `MelleaLogger` adds an OpenTelemetry
`LoggingHandler` alongside its existing handlers:

- **Console handler** — continues to work normally (color-coded output)
- **REST handler** — continues to work normally (when `MELLEA_LOG_WEBHOOK` is set)
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

1. Verify `MELLEA_LOG_OTLP=true` is set.
2. Check that an OTLP endpoint is configured
   (`OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_LOG_ENDPOINT`).
3. Verify the OTLP collector is running and configured to receive logs.
4. Check collector logs for connection errors.

**Warning about missing endpoint:**

```text
WARNING: OTLP logs exporter is enabled but no endpoint is configured
```

Set either `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_LOG_ENDPOINT`.

**Connection refused:**

1. Verify the OTLP collector is running: `docker ps | grep otel`
2. Check the endpoint URL is correct (default: `http://localhost:4317`).
3. Verify network connectivity: `curl http://localhost:4317`

---

**See also:**

- [Telemetry](../observability/telemetry) — overview of all
  telemetry features and configuration.
- [Tracing](../observability/tracing) — distributed traces
  with Gen-AI semantic conventions.
- [Metrics](../observability/metrics) — metrics, exporters,
  and custom instruments.
