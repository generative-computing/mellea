import mellea
from mellea.stdlib.base import SimpleContext
from mellea.stdlib.requirement import check, req, simple_validate
from mellea.stdlib.sampling.majority_voting import (
    MBRDRougeLStrategy,
    MajorityVotingStrategyForMath
)
import pytest


class TestMajorityVoting:

    def test_majority_voting_for_math(self):
        m = mellea.start_session(ctx=SimpleContext())
        query = "Compute 1+1"
        prompt_suffix = "\nPlease reason step by step, use \n\n to end each step, and put your final answer within \\boxed{}."
        prompt = query + prompt_suffix

        result = m.instruct(
            prompt,
            strategy=MajorityVotingStrategyForMath(number_of_samples=8, loop_budget=1),
            return_sampling_results=True,
        )
        if result.success:
            output = str(result.result)
        else:
            output = result.sample_generations[0].value

        print(output)
        assert output

    def test_MBRDRougeL(self):
        m = mellea.start_session(ctx=SimpleContext())
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
        email_candidate = m.instruct(
            "Write an email to {{name}} using the notes following: {{notes}}.",
            requirements=requirements,
            strategy=MBRDRougeLStrategy(number_of_samples=8, loop_budget=1),
            user_variables={"name": name, "notes": notes},
            return_sampling_results=True,
        )

        if email_candidate.success:
            output = str(email_candidate.result)
        else:
            output = email_candidate.sample_generations[0].value

        print(output)
        assert output


if __name__ == "__main__":
    pytest.main(["-s", __file__])
