from mellea.backends import ModelOption
from mellea import start_session, MelleaSession
from mellea.stdlib.requirements import check, req, simple_validate
from mellea.stdlib.sampling.majority_voting import (
    MBRDRougeLStrategy,
    MajorityVotingStrategyForMath,
)
import pytest

from mellea.core import SamplingResult


@pytest.fixture(scope="module")
def m_session(gh_run):
    if gh_run == 1:
        m = start_session(
            "ollama",
            model_id="llama3.2:1b",
            model_options={ModelOption.MAX_NEW_TOKENS: 5},
        )
    else:
        m = start_session("ollama", model_id="llama3.2:1b")
    yield m
    del m


def test_majority_voting_for_math(m_session: MelleaSession):
    query = "Compute 1+1"
    prompt_suffix = "\nPlease reason step by step, use \n\n to end each step, and put your final answer within \\boxed{}."
    prompt = query + prompt_suffix

    result = m_session.instruct(
        prompt,
        strategy=MajorityVotingStrategyForMath(number_of_samples=8, loop_budget=1),
        return_sampling_results=True,
    )
    output = str(result.result)

    print(output)
    assert output


def test_MBRDRougeL(m_session: MelleaSession):
    requirements = [
        req("The email should have a salutation"),  # == r1
        req(
            "Use only lower-case letters",
            validation_fn=simple_validate(lambda x: x.lower() == x),
        ),  # == r2
        check("Do not mention purple elephants."),  # == r3
    ]

    name = "Olivia"
    notes = "Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery."
    email_candidate: SamplingResult = m_session.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=requirements,  # type: ignore
        strategy=MBRDRougeLStrategy(number_of_samples=8, loop_budget=1),
        user_variables={"name": name, "notes": notes},
        return_sampling_results=True,
    )

    output = str(email_candidate.result)

    print(output)
    assert output


if __name__ == "__main__":
    pytest.main(["-s", __file__])
