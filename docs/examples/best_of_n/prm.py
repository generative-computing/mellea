"""Example of Using Best of N with PRMs."""

from docs.examples.helper import w
from mellea import start_session
from mellea.backends.process_reward_models.huggingface.prms import (
    HFGenerativePRM,
    HFRegressionPRM,
)
from mellea.backends.types import ModelOption
from mellea.stdlib.rewards.prm_scorer import PRMScorer
from mellea.stdlib.sampling.best_of_n import BestofNSamplingStrategy

# create a session for the generator using Granite 3.3 8B on Huggingface and a simple context [see below]
m = start_session(backend_name="hf", model_options={ModelOption.MAX_NEW_TOKENS: 512})

# initialize the PRM model
prm_model = HFGenerativePRM(
    model_name_or_path="ibm-granite/granite-3.3-8b-lora-math-prm",
    score_token="Y",
    generation_prompt="Is this response correct so far (Y/N)?",
    step_separator="\n\n",
)

# # can also initialize a Regression PRM model
# prm_model = HFRegressionPRM(
#     model_name_or_path = "granite-3.3-8b-math-prm-regression",
#     score_token= "<end_of_step>",
#     step_separator= "\n\n")

# create PRM scorer object
prm = PRMScorer(prm_model=prm_model, preference_ordering="max")

# Do Best of N sampling with the PRM scorer and an additional requirement
BoN_prm = m.instruct(
    "Sarah has 12 apples. She gives 5 of them to her friend. How many apples does Sarah have left?",
    strategy=BestofNSamplingStrategy(loop_budget=3),
    model_options={"temperature": 0.9, "do_sample": True},
    requirements=["provide final answer like 'Final Answer:'", prm],
)

# print result
print(f"***** BoN ****\n{w(BoN_prm)}\n*******")
