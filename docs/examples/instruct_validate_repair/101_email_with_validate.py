# pytest: ollama, e2e
"""Explicit validation and sampling inspection.

Demonstrates using `RejectionSamplingStrategy` with `return_sampling_results=True`
to inspect generation attempts, validation outcomes, and requirement satisfaction
reports across sampling loops.

Usage:
    uv run python docs/examples/instruct_validate_repair/101_email_with_validate.py
"""

from docs.examples.helper import req_print, w
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 200})

email_v2_samples = m.instruct(
    "Write a very short email to invite all interns to the office party.",
    requirements=["Use formal language.", "Use 'Dear Interns' as greeting."],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True,
)

if email_v2_samples.success:
    print(f"Success: \n{w(email_v2_samples.result)}")
    print(
        f"===> Requirement for this sample: \n{req_print(email_v2_samples.sample_validations[-1])}"
    )
else:
    print(f"Failure: \n{w(email_v2_samples.result)}")
    selected_index = email_v2_samples.sample_generations.index(email_v2_samples.result)
    print(
        f"===> Requirement for this sample: \n{req_print(email_v2_samples.sample_validations[selected_index])}"
    )
