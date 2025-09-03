"""Example of Using Best of N with PRMs"""

from docs.examples.helper import w
from mellea import start_session
from mellea.backends.types import ModelOption
from mellea.stdlib.rewards.prm_scorer import PRMScorer
from mellea.stdlib.sampling import BestofNSamplingStrategy

# create a session using Granite 3.3 8B on Huggingface and a simple context [see below]
m = start_session(backend_name="hf", model_options={ModelOption.MAX_NEW_TOKENS: 1024})

# create PRM scorer object
prm = PRMScorer(
    model_version="ibm-granite/granite-3.3-8b-lora-math-prm",
    prm_type="generative",
    correct_token="Y",
    generation_prompt="Is this response correct so far (Y/N)?",
    step_splitter="\n\n",
)

# Do Best of N sampling with the PRM scorer
BoN_prm = m.instruct(
    "Sarah has 12 apples. She gives 5 of them to her friend. How many apples does Sarah have left?",
    strategy=BestofNSamplingStrategy(loop_budget=3, requirements=[prm]),
    model_options={"temperature": 0.9, "do_sample": True},
)

# print result
print(f"***** BoN ****\n{w(BoN_prm)}\n*******")
