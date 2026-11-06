# OpenTelemetry Instrumentation Examples

This directory contains examples demonstrating OpenTelemetry tracing and metrics in Mellea.
For setup, exporters (including Jaeger), span attributes, and metrics, see
[Telemetry](../../docs/observability/telemetry.md),
[Tracing](../../docs/observability/tracing.md), and
[Metrics](../../docs/observability/metrics.md).

## Examples

| File | Description |
| :--- | :--- |
| [`telemetry_example.py`](./telemetry_example.py) | Demonstrates distributed tracing (application and backend traces). |
| [`metrics_example.py`](./metrics_example.py) | Demonstrates token usage metrics collection. |
| [`otel_genai_semconv_example.py`](./otel_genai_semconv_example.py) | Verifies OTel GenAI semantic convention attributes emitted on backend spans (`gen_ai.provider.name`, `gen_ai.conversation.id`, `error.type`). Designed for human verification against [otelite](https://github.com/planetf1/otelite). |

## Performance Impact

- **Disabled (default)**: Near-zero overhead
- **Enabled**: Low overhead (~2-5%)

## Troubleshooting

**Traces not appearing in Jaeger:**
1. Check Jaeger is running: `docker ps | grep jaeger`
2. Verify endpoint: `curl http://localhost:4317`
3. Check environment variables are set
4. Enable console output to verify traces are created
