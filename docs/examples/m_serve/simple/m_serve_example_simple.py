# pytest: ollama, e2e

"""Example to run m serve.

Demonstrates structuring a Mellea program for serving as an API with
`m serve`: defines a `serve()` function that applies two `Requirement`
constraints and a `RejectionSamplingStrategy` before returning.

Prerequisites:
    ollama serve
    ollama pull granite4.2:3b

Usage:
    m serve docs/examples/m_serve/simple/m_serve_example_simple.py
"""

import mellea
from mellea.core import ModelOutputThunk, Requirement, SamplingResult
from mellea.serve import ChatMessage
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy


def validate_hi_bob(email: str) -> bool:
    return email.startswith("Hi Bob!")


def validate_email_len(email: str) -> bool:
    return len(email) < 50


def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: None | dict = None,
) -> ModelOutputThunk | SamplingResult:
    """Takes a prompt as input and runs it through an M program."""
    requirements = requirements if requirements else []
    message = input[-1].content
    reqs = [
        Requirement(
            "Keep this under 50 words",
            validation_fn=simple_validate(validate_email_len),
        ),
        Requirement(
            "Add a 'Hi Bob!' at the top of the output",
            validation_fn=simple_validate(validate_hi_bob),
        ),
        *requirements,
    ]

    session = mellea.start_session()
    result = session.instruct(
        description=message,  # type: ignore
        requirements=reqs,  # type: ignore
        strategy=RejectionSamplingStrategy(loop_budget=3),
        model_options=model_options,
    )
    return result
