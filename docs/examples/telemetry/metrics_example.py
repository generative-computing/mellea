# pytest: ollama, e2e

"""Example demonstrating OpenTelemetry metrics exporters in Mellea.

This example shows how to use metrics with different exporters:
- Console: Print metrics to console for debugging
- OTLP: Export to OpenTelemetry Protocol collectors
- Prometheus: Expose HTTP endpoint for Prometheus scraping

Run with different configurations:

# 1. Console exporter (simplest - for debugging)
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
python metrics_example.py

# 2. OTLP exporter (production observability)
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_OTLP=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=mellea-metrics-example
python metrics_example.py

# 3. Prometheus exporter (Prometheus monitoring)
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_PROMETHEUS=true
python metrics_example.py
# Then access metrics at: curl http://localhost:9464/metrics

# 4. Multiple exporters simultaneously
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
export MELLEA_METRICS_OTLP=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export MELLEA_METRICS_PROMETHEUS=true
python metrics_example.py

# For OTLP Collector and Prometheus setup instructions, see:
# docs/docs/evaluation-and-observability/metrics.md
"""

import os

from mellea import generative, start_session
from mellea.backends import ModelOption
from mellea.stdlib.requirements import req


@generative
def summarize_text(text: str) -> str:
    """Summarize the given text in one sentence."""


@generative
def translate_to_spanish(text: str) -> str:
    """Translate the given text to Spanish."""


def main():
    """Run example with metrics collection."""
    print("=" * 60)
    print("Mellea Metrics Example")
    print("=" * 60)

    # Check if metrics are enabled
    from mellea.telemetry import is_metrics_enabled

    if not is_metrics_enabled():
        print("⚠️  Metrics are disabled!")
        print("Enable with: export MELLEA_METRICS_ENABLED=true")
        print("=" * 60)
        return

    print("Metrics enabled")

    # When Prometheus is enabled, start an HTTP server to expose metrics
    if os.getenv("MELLEA_METRICS_PROMETHEUS", "false").lower() == "true":
        from prometheus_client import start_http_server

        start_http_server(9464)
        print("✓ Prometheus endpoint: http://localhost:9464/metrics")

    print("=" * 60)

    # Start a session - metrics recorded automatically
    with start_session() as m:
        # Example 1: Simple generation
        print("\n1. Simple generation...")
        summary = summarize_text(
            m,
            text="Artificial intelligence is transforming how we work, learn, and interact with technology. "
            "From healthcare to education, AI systems are becoming increasingly sophisticated and accessible.",
        )
        print(f"Summary: {summary}")

        # Example 2: Generation with requirements
        print("\n2. Generation with requirements...")
        email = m.instruct(
            "Write a brief email to {{name}} about {{topic}}",
            requirements=[req("Must be under 50 words"), req("Must be professional")],
            user_variables={"name": "Dr. Smith", "topic": "meeting schedule"},
        )
        print(f"Email: {str(email)[:100]}...")

        # Token usage is available on the result from instruct()
        if email.usage:
            print(f"  → Prompt tokens: {email.usage['prompt_tokens']}")
            print(f"  → Completion tokens: {email.usage['completion_tokens']}")
            print(f"  → Total tokens: {email.usage['total_tokens']}")

        # Example 3: Multiple operations
        print("\n3. Multiple operations...")
        text = "Hello, how are you today?"
        translation = translate_to_spanish(m, text=text)
        print(f"Translation: {translation}")

        # Example 4: Chat interaction
        print("\n4. Chat interaction...")
        response = m.chat("What is the capital of France?")
        print(f"Response: {str(response)[:100]}...")

        # Example 5: Streaming with latency metrics
        print("\n5. Streaming request (latency metrics)...")
        streamed = m.instruct(
            "Name three programming languages in one sentence.",
            model_options={ModelOption.STREAM: True},
        )
        print(f"Response: {str(streamed)[:100]}...")
        if streamed.streaming and streamed.ttfb_ms is not None:
            print(f"  -> Time to first token: {streamed.ttfb_ms:.1f} ms")

    print("\n" + "=" * 60)
    print("Example complete! Metrics recorded.")

    # When Prometheus is enabled, keep the process running so the endpoint can be scraped
    if os.getenv("MELLEA_METRICS_PROMETHEUS", "false").lower() == "true":
        print("Prometheus endpoint still available at http://localhost:9464/metrics")
        print("Press Ctrl+C to exit.")
        print("=" * 60)

        import time

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down.")
    else:
        print("=" * 60)


if __name__ == "__main__":
    main()
