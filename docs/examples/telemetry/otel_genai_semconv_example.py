# pytest: ollama, e2e, qualitative

"""Mellea backend spans carrying OTel GenAI semantic convention attributes.

Each backend generation call emits a ``chat`` span with the following attributes
drawn from the OTel GenAI semconv (https://opentelemetry.io/docs/specs/semconv/gen-ai/):

  gen_ai.provider.name   — provider identity (current semconv)
  gen_ai.system          — same value, retained for back-compat with existing dashboards
  gen_ai.conversation.id — correlated to the active session via ``with_context``
  error.type             — set on the error path alongside ERROR span status

Run against otelite for human verification:

  # Terminal 1 — start otelite (OTLP gRPC :4317, UI :8080)
  docker run --rm -p 4317:4317 -p 8080:8080 ghcr.io/planetf1/otelite:latest

  # Terminal 2
  export MELLEA_TRACE_BACKEND=1
  export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
  export OTEL_SERVICE_NAME=mellea-semconv-demo
  python otel_genai_semconv_example.py

  Then open http://localhost:8080 and select the mellea-semconv-demo service.

Expected span attributes
------------------------
  Span "chat" (normal path)
    gen_ai.system              = "ollama"
    gen_ai.provider.name       = "ollama"
    gen_ai.conversation.id     = "demo-session-1"
    mellea.session_id          = "demo-session-1"

  Span "chat" (error path)
    error.type  = <exception class name>
    status      = ERROR
"""

from mellea import start_session
from mellea.telemetry import is_backend_tracing_enabled, with_context


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def main() -> None:
    _section("Mellea OTel GenAI Semantic Convention Demo")
    print(f"Backend tracing: {is_backend_tracing_enabled()}")
    if not is_backend_tracing_enabled():
        print("Set MELLEA_TRACE_BACKEND=1 to enable backend spans.")

    # -----------------------------------------------------------------------
    # Normal path: provider name + conversation id
    # -----------------------------------------------------------------------
    _section("Normal path — provider name and conversation id")
    print("Expected span attrs:")
    print("  gen_ai.system              = 'ollama'")
    print("  gen_ai.provider.name       = 'ollama'")
    print("  gen_ai.conversation.id     = 'demo-session-1'")

    with with_context(session_id="demo-session-1"):
        with start_session() as m:
            result = m.instruct("Summarise quantum tunnelling in one sentence.")
    print(f"\nOutput: {str(result)[:120]}")

    # -----------------------------------------------------------------------
    # Error path: error.type + ERROR status
    # -----------------------------------------------------------------------
    _section("Error path — error.type on span")
    print("Expected span attrs:")
    print("  status     = ERROR")
    print("  error.type = <exception class name>")

    try:
        with start_session(base_url="http://localhost:19999") as m2:
            m2.instruct("Hello")
    except Exception as exc:
        print(f"\nGot expected error: {exc.__class__.__name__}")
    else:
        print("\n(No error — nothing is listening on port 19999)")

    _section("Done")
    print("If OTEL_EXPORTER_OTLP_ENDPOINT is set, check your trace backend.")
    print("If MELLEA_TRACE_CONSOLE=1, spans were printed to stdout above.")


if __name__ == "__main__":
    main()
