import mellea

# INFO: this line will download IBM's Granite 3.3 8B model.
m = mellea.start_session()

print("Basic email:")
email = m.instruct("Write an email inviting interns to an office party at 3:30pm.")
print(str(email))

print("Function with user variables:")


def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        user_variables={"name": name, "notes": notes},
    )
    return email.value  # str(email) also works.


print(
    write_email(
        m,
        "Olivia",
        "Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery.",
    )
)

print("Email with requirements:")


def write_email_with_requirements(
    m: mellea.MelleaSession, name: str, notes: str
) -> str:
    email = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=[
            "The email should have a salutation",
            "Use only lower-case letters",
        ],
        user_variables={"name": name, "notes": notes},
    )
    return str(email)


print(
    write_email_with_requirements(
        m,
        name="Olivia",
        notes="Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery.",
    )
)

print("Email with rejection sampling:")
from mellea.stdlib.sampling import RejectionSamplingStrategy  # noqa: E402


def write_email_with_strategy(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email_candidate = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=[
            "The email should have a salutation",
            "Use only lower-case letters",
        ],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        user_variables={"name": name, "notes": notes},
        return_sampling_results=True,
    )
    if email_candidate.success:
        return str(email_candidate.result)
    else:
        print("Expect sub-par result.")
        return email_candidate.sample_generations[0].value


print(
    write_email_with_strategy(
        m,
        "Olivia",
        "Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery.",
    )
)
