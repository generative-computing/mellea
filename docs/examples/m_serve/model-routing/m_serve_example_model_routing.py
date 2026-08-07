# pytest: ollama, e2e

"""Example showing how to use client_options to route on the client model ID.

In an OpenAI-compatible request the client sends a `model` string, e.g.:

    client.chat.completions.create(model="granite4.1:8b", messages=[...])

That string is **routing / metadata** from the server's perspective: `m serve`
echoes it back in the response but does NOT include it in `model_options` (which
is filtered for backend consumption).

Declare `client_options` in `serve()` and `m serve` passes the full raw client
request as a dict, giving access to `model` and every other field the client
sent — without any of those values leaking into `model_options`.

To ignore the client model ID entirely and always use a fixed backend, simply
omit the `client_options` parameter (see the simple/ examples).

Run the server:
    m serve docs/examples/m_serve/model-routing/m_serve_example_model_routing.py

Test with the client:
    python docs/examples/m_serve/model-routing/client_model_routing.py
"""

import os
from typing import Any

import mellea
from mellea.backends.model_ids import IBM_GRANITE_4_1_3B, IBM_GRANITE_4_1_8B
from mellea.core import ModelOutputThunk
from mellea.serve import ChatMessage

_ollama_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
if not _ollama_host.startswith(("http://", "https://")):
    _ollama_host = f"http://{_ollama_host}"

_DEFAULT_MODEL = IBM_GRANITE_4_1_3B

_ALLOWED_MODELS: dict[str, Any] = {
    IBM_GRANITE_4_1_3B.ollama_name: IBM_GRANITE_4_1_3B,  # type: ignore[dict-item]
    IBM_GRANITE_4_1_8B.ollama_name: IBM_GRANITE_4_1_8B,  # type: ignore[dict-item]
}


def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict[str, Any] | None = None,
    client_options: dict[str, Any] | None = None,
) -> ModelOutputThunk:
    """Serve with backend selected from the standard client `model` field.

    Reads `client_options["model"]` (the standard OpenAI `model` field) and
    routes to an allowlisted Ollama backend.  Falls back to `granite4.1:3b`
    when the value is unrecognised.  `model_options` is clean — it contains
    only backend generation parameters, never routing metadata.

    Args:
        input: Chat messages from the client.
        requirements: Optional requirement strings forwarded from the client.
        model_options: Generation parameters filtered for backend consumption.
        client_options: Full raw client request fields, including `model`.

    Returns:
        ModelOutputThunk with the generated response.
    """
    model_name = (client_options or {}).get("model")
    chosen_model = _ALLOWED_MODELS.get(model_name, _DEFAULT_MODEL)  # type: ignore[arg-type]

    message = input[-1].get_text_content() or "No message provided"
    session = mellea.start_session(model_id=chosen_model)
    return session.instruct(
        description=message,
        requirements=requirements,  # type: ignore[arg-type]
        model_options=model_options,
    )
