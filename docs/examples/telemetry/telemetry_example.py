"""Example demonstrating OpenTelemetry tracing in Mellea.

This example shows how to use the two independent trace scopes:
1. Application trace - tracks user-facing operations
2. Backend trace - tracks LLM backend interactions

Run with different configurations:

# Enable only application tracing
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=false
python telemetry_example.py

# Enable only backend tracing
export MELLEA_TRACE_APPLICATION=false
export MELLEA_TRACE_BACKEND=true
python telemetry_example.py

# Enable both traces
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=true
python telemetry_example.py

# Export to OTLP endpoint (e.g., Jaeger)
export MELLEA_TRACE_APPLICATION=true
export MELLEA_TRACE_BACKEND=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
python telemetry_example.py

# Enable console output for debugging
export MELLEA_TRACE_CONSOLE=true
python telemetry_example.py
"""

from mellea import generative, start_session
from mellea.stdlib.requirements import req


@generative
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of the given text as positive, negative, or neutral."""
    ...


@generative
def extract_entities(text: str) -> list[str]:
    """Extract named entities from the text."""
    ...


def main():
    """Run example with telemetry instrumentation."""
    print("=" * 60)
    print("Mellea OpenTelemetry Example")
    print("=" * 60)

    # Check which traces are enabled
    from mellea.telemetry import (
        is_application_tracing_enabled,
        is_backend_tracing_enabled,
    )

    print(f"Application tracing: {is_application_tracing_enabled()}")
    print(f"Backend tracing: {is_backend_tracing_enabled()}")
    print("=" * 60)

    # Start a session - this will be traced if application tracing is enabled
    with start_session() as m:
        # Example 1: Simple instruction with requirements
        print("\n1. Simple instruction with requirements...")
        email = m.instruct(
            "Write a professional email to {{name}} about {{topic}}",
            requirements=[req("Must be formal"), req("Must be under 100 words")],
            user_variables={"name": "Alice", "topic": "project update"},
        )
        print(f"Generated email: {str(email)[:100]}...")

        # Example 2: Using @generative function
        print("\n2. Using @generative function...")
        sentiment = classify_sentiment(
            m, text="I absolutely love this product! It's amazing!"
        )
        print(f"Sentiment: {sentiment}")

        # Example 3: Multiple operations
        print("\n3. Multiple operations...")
        text = "Apple Inc. announced new products in Cupertino, California."
        entities = extract_entities(m, text=text)
        print(f"Entities: {entities}")

        # Example 4: Chat interaction
        print("\n4. Chat interaction...")
        response1 = m.chat("What is 2+2?")
        print(f"Response 1: {str(response1)}")

        response2 = m.chat("Multiply that by 3")
        print(f"Response 2: {str(response2)}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nTrace data has been exported based on your configuration.")
    print("If OTEL_EXPORTER_OTLP_ENDPOINT is set, check your trace backend.")
    print("If MELLEA_TRACE_CONSOLE=true, traces are printed above.")


if __name__ == "__main__":
    main()
