"""Example to run m serve."""

from typing import Any

import mellea
from cli.serve.models import ChatMessage
from mellea.stdlib.base import ChatContext, ModelOutputThunk
from mellea.stdlib.requirement import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy, SamplingResult

session = mellea.start_session(ctx=ChatContext())


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

    result = session.instruct(
        description=message,  # type: ignore
        requirements=reqs,  # type: ignore
        strategy=RejectionSamplingStrategy(loop_budget=3),
        model_options=model_options,
    )
    return result
