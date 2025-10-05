"""Advanced Example of information extraction with Mellea using m.instruct() and constraints."""

from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from mellea import start_session
from mellea.backends import model_ids
from mellea.stdlib.requirement import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy, SamplingResult

# ref: https://www.nytimes.com/2012/05/20/world/world-leaders-at-us-meeting-urge-growth-not-austerity.html
NYTimes_text = "CAMP DAVID, Md. — Leaders of the world's richest countries banded together on Saturday to press Germany to back more pro-growth policies to halt the deepening debt crisis in Europe, as President Obama for the first time gained widespread support for his argument that Europe, and the United States by extension, cannot afford Chancellor Angela Merkel's one-size-fits-all approach emphasizing austerity."


# pydantic object for output formatting
class NameResponse(BaseModel):
    names: list[str]


# a function to check that output is parsable (which it should) and
# checks that the output contains at least n names
def at_least_n(n: int) -> Callable[[str], bool]:
    def _at_least_(t: str) -> bool:
        try:
            nr = NameResponse.model_validate_json(t)
            if len(nr.names) >= n:
                return True
            else:
                return False
        except ValidationError:
            return False

    return _at_least_


# start session
m = start_session()

# run extraction using grounding context and sampling strategy
sampled_p_names = m.instruct(
    "Extract ALL person names from the document (doc1).",
    grounding_context={"doc1": NYTimes_text},
    requirements=[check(None, validation_fn=simple_validate(at_least_n(2)))],
    strategy=RejectionSamplingStrategy(loop_budget=5),
    format=NameResponse,
    return_sampling_results=True,
)

assert isinstance(sampled_p_names, SamplingResult)

# if sampling has been a success (all requirements are met)...
if sampled_p_names.success:
    person_names = NameResponse.model_validate_json(str(sampled_p_names.result)).names
else:
    person_names = ["LLM call did not yield a result."]

print(f"person_names = {person_names}")
# out ~: person_names = ['President Obama', 'Angela Merkel']
