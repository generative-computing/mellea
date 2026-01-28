from mellea.stdlib.requirements import check, req, simple_validate

requirements = [
    req("The email should have a salutation"),  # == r1
    req(
        "Use only lower-case letters",
        validation_fn=simple_validate(lambda x: x.lower() == x),
    ),  # == r2
    check("Do not mention purple elephants."),  # == r3
]

import mellea
from mellea.stdlib.sampling import RejectionSamplingStrategy


def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email_candidate = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=requirements,
        strategy=RejectionSamplingStrategy(loop_budget=5),
        user_variables={"name": name, "notes": notes},
        return_sampling_results=True,
    )
    if email_candidate.success:
        return str(email_candidate.result)
    else:
        return email_candidate.sample_generations[0].value


m = mellea.start_session()
print(
    write_email(
        m,
        "Olivia",
        "Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery.",
    )
)
