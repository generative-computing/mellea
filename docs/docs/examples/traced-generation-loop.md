---
title: "Traced Generation Loop"
description: "Enable OpenTelemetry tracing for a multi-operation Mellea session using environment variables, and export spans to Jaeger or any OTLP backend."
# diataxis: reference
---

This example runs a session that exercises four different Mellea operations —
`m.instruct()`, a `@generative` classifier, a `@generative` entity extractor,
and a multi-turn `m.chat()` — while OpenTelemetry instrumentation records each
step. Two trace scopes are populated: the application trace covers Mellea-level
operations, and the backend trace covers raw LLM calls.

**Source file:** `docs/examples/telemetry/telemetry_example.py`

## Concepts covered

- The two trace scopes: `mellea.application` and `mellea.backend`
- Enabling tracing with `MELLEA_TRACES_ENABLED`
- Using `start_session()` as a context manager so session lifecycle is spanned
- Exporting spans to an OTLP endpoint (Jaeger)
- Using `mellea.stdlib.requirements.req` to attach constraints to `m.instruct()`

## Prerequisites

- [Quick Start](../getting-started/quickstart) complete
- Ollama running locally with `granite4.1:3b` pulled
- (Optional) [Jaeger](https://www.jaegertracing.io/) running locally for span
  visualisation — see the Jaeger section below

Install with all extras to get the OpenTelemetry dependencies:

```bash
uv sync --all-extras
```

## Trace scopes

Mellea defines two OpenTelemetry trace scopes.

| Scope | What it records |
| ----- | --------------- |
| Application (`mellea.application`) | Session lifecycle, `@generative` calls, `aact`, sampling, requirement validation |
| Backend (`mellea.backend`) | Raw model generation calls, context-based generation, backend-specific operations |

### Performance impact

| Configuration | Overhead |
| ------------- | -------- |
| Disabled (default) | Near-zero |
| Enabled | ~2–5 % |

## Running the example

### No tracing (baseline)

```bash
python docs/examples/telemetry/telemetry_example.py
```

### Tracing enabled with console output for debugging

```bash
export MELLEA_TRACES_ENABLED=true
export MELLEA_TRACES_CONSOLE=true
python docs/examples/telemetry/telemetry_example.py
```

### Export to an OTLP endpoint (Jaeger)

```bash
export MELLEA_TRACES_ENABLED=true
export MELLEA_TRACES_OTLP=true
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
python docs/examples/telemetry/telemetry_example.py
```

## Starting Jaeger

Run Jaeger in Docker to receive and visualise spans:

```bash
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

After running the example, open `http://localhost:16686`, select the
`mellea-example` service, and browse the trace timeline.

## The full example

### Generative function declarations

```python
from mellea import generative, start_session
from mellea.stdlib.requirements import req


@generative
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of the given text as positive, negative, or neutral."""


@generative
def extract_entities(text: str) -> list[str]:
    """Extract named entities from the text."""
```

These two functions are declared at module level. `@generative` wires them up
to the runtime; no implementation is needed. Each call site below passes a
session `m` as the first argument, which binds the call to the current trace
context.

### Session as a context manager and introspection

```python
from mellea.telemetry import is_tracing_enabled


def main():
    """Run example with telemetry instrumentation."""
    print("=" * 60)
    print("Mellea OpenTelemetry Example")
    print("=" * 60)

    print(f"Tracing enabled: {is_tracing_enabled()}")
    print("=" * 60)
```

`is_tracing_enabled()` reflects the current environment variable state at
runtime. Use this guard in your own code when you want to conditionally add
tracing context (for example, adding custom span attributes only when tracing
is on).

### Operation 1: instruct with requirements

```python
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
```

Using `start_session()` as a context manager (`with start_session() as m:`)
means the session open and close events are recorded as the root span when
application tracing is enabled. All child operations appear nested under this
root.

`req("Must be formal")` attaches a soft requirement to the generation.
Requirements appear as span attributes in the trace so you can see which
constraints were applied and whether they triggered a retry.

### Operation 2: @generative sentiment classifier

```python
        # Example 2: Using @generative function
        print("\n2. Using @generative function...")
        sentiment = classify_sentiment(
            m, text="I absolutely love this product! It's amazing!"
        )
        print(f"Sentiment: {sentiment}")
```

Each `@generative` call produces its own child span in the application trace.
The span includes the function name, parameter names, and the inferred return
type.

### Operation 3: @generative entity extractor

```python
        # Example 3: Multiple operations
        print("\n3. Multiple operations...")
        text = "Apple Inc. announced new products in Cupertino, California."
        entities = extract_entities(m, text=text)
        print(f"Entities: {entities}")
```

Running multiple `@generative` calls inside the same `with` block keeps them
all under the same root span. In Jaeger you can see the sequence and duration of
each call on a single timeline.

### Operation 4: multi-turn chat

```python
        # Example 4: Chat interaction
        print("\n4. Chat interaction...")
        response1 = m.chat("What is 2+2?")
        print(f"Response 1: {response1!s}")

        response2 = m.chat("Multiply that by 3")
        print(f"Response 2: {response2!s}")
```

`m.chat()` is a stateful multi-turn method. The session accumulates turn
history, so `response2` can refer back to the result of `response1` without
repeating the context. Both turns appear as sibling spans under the root session
span.

### Full file

```python
# pytest: ollama, e2e

"""Example demonstrating OpenTelemetry tracing in Mellea.

This example shows the two trace scopes populated when tracing is enabled:
1. Application trace - tracks user-facing operations
2. Backend trace - tracks LLM backend interactions

Run with different configurations:

# Enable tracing
export MELLEA_TRACES_ENABLED=true
python telemetry_example.py

# Export to OTLP endpoint (e.g., Jaeger)
export MELLEA_TRACES_ENABLED=true
export MELLEA_TRACES_OTLP=true
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
python telemetry_example.py

# Enable console output for debugging
export MELLEA_TRACES_ENABLED=true
export MELLEA_TRACES_CONSOLE=true
python telemetry_example.py
"""

from mellea import generative, start_session
from mellea.stdlib.requirements import req
from mellea.telemetry import is_tracing_enabled


@generative
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of the given text as positive, negative, or neutral."""


@generative
def extract_entities(text: str) -> list[str]:
    """Extract named entities from the text."""


def main():
    """Run example with telemetry instrumentation."""
    print("=" * 60)
    print("Mellea OpenTelemetry Example")
    print("=" * 60)

    print(f"Tracing enabled: {is_tracing_enabled()}")
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
        print(f"Response 1: {response1!s}")

        response2 = m.chat("Multiply that by 3")
        print(f"Response 2: {response2!s}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nTrace data has been exported based on your configuration.")
    print("If OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is set, check your trace backend.")
    print("If MELLEA_TRACES_CONSOLE=true, traces are printed above.")


if __name__ == "__main__":
    main()
```

## Span attributes

Each span in the application trace includes the following attributes where
applicable:

| Attribute | Description |
| --------- | ----------- |
| `mellea.model_id` | Model identifier used for the call |
| `mellea.backend` | Backend identifier (e.g. `"ollama"`) |
| `mellea.action_type` | Component type (e.g. `generative`, `instruct`) |
| `mellea.context_size` | Number of context items passed |
| `mellea.has_requirements` | Whether requirements were specified |
| `mellea.strategy_type` | Sampling strategy used |
| `mellea.tool_calls` | Whether tool calling was enabled |
| `mellea.format_type` | Response format class |

## What to try next

- Set `OTEL_SERVICE_NAME=my-app` to customise the service name in your trace
  backend.
- See [Tracing](../observability/tracing)
  for attribute schemas and advanced configuration.
- Add `MELLEA_TRACES_CONSOLE=true` alongside an OTLP endpoint to confirm spans
  are generated even when the remote collector is unavailable.
