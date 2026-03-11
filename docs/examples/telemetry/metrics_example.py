# pytest: ollama, llm

"""Example demonstrating OpenTelemetry metrics in Mellea.

This example shows how to use token usage metrics to track LLM consumption.

Run with different configurations:

# Enable metrics with console output (simplest)
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
python metrics_example.py

# Enable metrics with OTLP export (requires OTLP collector)
export MELLEA_METRICS_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=mellea-metrics-example
python metrics_example.py
"""

from mellea import generative, start_session
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
    print("Mellea Token Metrics Example")
    print("=" * 60)

    # Check if metrics are enabled
    from mellea.telemetry import is_metrics_enabled

    if not is_metrics_enabled():
        print("⚠️  Metrics are disabled!")
        print("Enable with: export MELLEA_METRICS_ENABLED=true")
        print("=" * 60)
        return

    print("✓ Token metrics enabled")
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

        # Example 3: Multiple operations
        print("\n3. Multiple operations...")
        text = "Hello, how are you today?"
        translation = translate_to_spanish(m, text=text)
        print(f"Translation: {translation}")

        # Example 4: Chat interaction
        print("\n4. Chat interaction...")
        response = m.chat("What is the capital of France?")
        print(f"Response: {str(response)[:100]}...")

    print("\n" + "=" * 60)
    print("Example complete! Token metrics recorded.")
    print("=" * 60)


if __name__ == "__main__":
    main()
