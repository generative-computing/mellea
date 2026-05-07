# pytest: ollama, e2e

"""Example demonstrating response_format with m serve.

This example shows how to use the response_format parameter to get structured
output from the model. The server supports three format types:
- text: Plain text output (default)
- json_object: Unstructured JSON output
- json_schema: Structured output validated against a JSON schema

Run the server:
    m serve docs/examples/m_serve/m_serve_example_response_format.py

Test with the client:
    python docs/examples/m_serve/client_response_format.py
"""

from typing import Any

import mellea
from cli.serve.models import ChatMessage
from mellea.core import ModelOutputThunk
from mellea.stdlib.context import ChatContext

session = mellea.start_session(ctx=ChatContext())


def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict[str, Any] | None = None,
    format: type | None = None,
) -> ModelOutputThunk:
    """Serve function that supports response_format parameter.

    Args:
        input: List of chat messages from the client
        requirements: Optional list of requirement strings
        model_options: Optional model configuration parameters
        format: Optional Pydantic model for structured output (from response_format)

    Returns:
        ModelOutputThunk with the generated response
    """
    message = input[-1].content or "No message provided"

    # When format is provided (from json_schema response_format),
    # pass it to instruct() to get structured output
    result = session.instruct(
        description=message,
        requirements=requirements,  # type: ignore
        model_options=model_options,
        format=format,  # This enables structured output validation
    )

    return result
