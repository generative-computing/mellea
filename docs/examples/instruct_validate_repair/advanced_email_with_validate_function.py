# pytest: ollama, e2e
"""Custom validation functions and sampling strategies.

Demonstrates creating custom validation requirements using `simple_validate()`
and enforcing them with `RejectionSamplingStrategy` to ensure length constraints.

Usage:
    uv run python docs/examples/instruct_validate_repair/advanced_email_with_validate_function.py
"""

from docs.examples.helper import w
from mellea import start_session
from mellea.backends import ModelOption
from mellea.core import Requirement
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 200})

word_limit_req = Requirement(
    "Use less than 100 words.",
    validation_fn=simple_validate(lambda output: len(output.split(" ")) < 100),
)

email_v1 = m.instruct(
    "Write an email to invite all interns to the office party.",
    requirements=["be formal", word_limit_req],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)

print(f"***** email ****\n{w(email_v1)}\n*******")
