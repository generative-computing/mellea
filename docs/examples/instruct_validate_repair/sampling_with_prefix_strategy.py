from mellea.backends import ModelOption
from mellea.stdlib.requirement import check, req, simple_validate
from mellea.stdlib.sampling.prefix_cached import RejectionSamplingStrategyWithPrefix

requirements = [
    req("The email should have a salutation"),  # == r1
    req(
        "Use only lower-case letters",
        validation_fn=simple_validate(lambda x: x.lower() == x),
    ),  # == r2
    check("Do not mention purple elephants."),  # == r3
    req("The email should be funny."),
]

import mellea  # noqa: E402


def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email_candidate = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=requirements,
        strategy=RejectionSamplingStrategyWithPrefix(loop_budget=5),
        user_variables={"name": name, "notes": notes},
        return_sampling_results=True,
    )

    if email_candidate.success:
        return str(email_candidate.result)
    else:
        return email_candidate.sample_generations[0].value
