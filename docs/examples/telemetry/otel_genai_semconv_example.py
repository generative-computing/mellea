# pytest: ollama, e2e

"""Example demonstrating OTel GenAI semantic convention attributes (issue #1035).

Exercises the five emission-gap fixes added in this issue so they can be verified
in otelite or any OTel-compatible backend:

  gen_ai.provider.name      — provider identity (alongside legacy gen_ai.system)
  gen_ai.conversation.id    — mapped from session_id ContextVar
  llm.prompt_template.*     — template text (always) and variables (opt-in)
  error.type                — set on the error path alongside ERROR status
  gen_ai.input/output.messages — structured content (opt-in via MELLEA_TRACE_CONTENT)

Run against otelite for human verification:

  # Terminal 1 — start otelite (OTLP gRPC :4317, UI :8080)
  docker run --rm -p 4317:4317 -p 8080:8080 ghcr.io/planetf1/otelite:latest

  # Terminal 2 — run with all attributes visible
  export MELLEA_TRACE_BACKEND=1
  export MELLEA_TRACE_CONTENT=1
  export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
  export OTEL_SERVICE_NAME=mellea-semconv-demo
  python otel_genai_semconv_example.py

  Then open http://localhost:8080 → select mellea-semconv-demo service.

What to verify per span in otelite
-----------------------------------
  Span "chat"
    gen_ai.system              = "ollama"     (back-compat)
    gen_ai.provider.name       = "ollama"     (new, semconv v1.37.0)
    gen_ai.conversation.id     = "demo-session-1"
    mellea.session_id          = "demo-session-1"  (preserved)
    llm.prompt_template.template = "Summarise {{topic}} in one sentence."
    llm.prompt_template.variables = {"topic": "quantum tunnelling"}  (only with MELLEA_TRACE_CONTENT)
    gen_ai.input.messages      = [...]        (only with MELLEA_TRACE_CONTENT)
    gen_ai.output.messages     = [...]        (only with MELLEA_TRACE_CONTENT)

  Span "chat" (error path)
    error.type  = "OllamaRequestError" (or similar)
    status      = ERROR
"""

from mellea import start_session
from mellea.telemetry import (
    is_backend_tracing_enabled,
    is_content_tracing_enabled,
    with_context,
)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    _section("Mellea OTel GenAI Semantic Convention Demo")
    print(f"Backend tracing:  {is_backend_tracing_enabled()}")
    print(f"Content capture:  {is_content_tracing_enabled()}")
    if not is_backend_tracing_enabled():
        print("\nSet MELLEA_TRACE_BACKEND=1 to enable backend spans.")

    # -----------------------------------------------------------------------
    # 1. Provider name + conversation id + prompt template attrs
    # -----------------------------------------------------------------------
    _section("1. Provider name / conversation id / template attrs")
    print("Expected span attrs:")
    print("  gen_ai.system              = 'ollama'")
    print("  gen_ai.provider.name       = 'ollama'")
    print("  gen_ai.conversation.id     = 'demo-session-1'")
    print("  llm.prompt_template.template = 'Summarise {{topic}} in one sentence.'")

    with with_context(session_id="demo-session-1"):
        with start_session() as m:
            result = m.instruct(
                "Summarise {{topic}} in one sentence.",
                user_variables={"topic": "quantum tunnelling"},
            )
    print(f"\nOutput: {str(result)[:120]}")

    # -----------------------------------------------------------------------
    # 2. Content capture (opt-in)
    # -----------------------------------------------------------------------
    if is_content_tracing_enabled():
        _section("2. Content capture (MELLEA_TRACE_CONTENT=1)")
        print("Expected span attrs:")
        print("  gen_ai.system_instructions — serialised system turns")
        print("  gen_ai.input.messages      — [{'role':'user','parts':[...]}]")
        print(
            "  gen_ai.output.messages     — [{'role':'assistant','parts':[...],'finish_reason':'stop'}]"
        )
        print("  llm.prompt_template.variables = {'name': 'Ada'}")

        with start_session() as m2:
            result2 = m2.instruct(
                "Write a one-line greeting for {{name}}.",
                user_variables={"name": "Ada"},
            )
        print(f"\nOutput: {str(result2)[:120]}")
    else:
        _section("2. Content capture (skipped — set MELLEA_TRACE_CONTENT=1)")

    # -----------------------------------------------------------------------
    # 3. Error path: error.type + ERROR status
    # -----------------------------------------------------------------------
    _section("3. Error path — error.type on span")
    print("Expected span attrs:")
    print("  status     = ERROR")
    print("  error.type = <exception class name>")

    try:
        with start_session() as m3:
            # Use a model name guaranteed to be absent on any Ollama instance.
            m3._backend.model_id = "mellea-semconv-nonexistent-xyz"  # type: ignore[attr-defined]
            m3.instruct("Hello")
    except Exception as exc:
        print(f"\nGot expected error: {exc.__class__.__name__}")
    else:
        print(
            "\n(No error — check the span for error.type if the model unexpectedly exists)"
        )

    _section("Done")
    print("If OTEL_EXPORTER_OTLP_ENDPOINT is set, check your trace backend.")
    print("If MELLEA_TRACE_CONSOLE=1, spans were printed to stdout above.")


if __name__ == "__main__":
    main()
