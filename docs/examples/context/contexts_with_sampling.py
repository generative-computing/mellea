# pytest: ollama, llm

from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.session import start_session

# You can retrieve context information when using SamplingStrategies
# and validation.

m = start_session()

# We want the full SamplingResult.
res = m.instruct(
    "Write a sentence.",
    requirements=["be funny", "be formal", "start the sentence with the letter w"],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True,
)

print()
print("Printing result of `Writing a sentence`.")
print(f"Result: {res.success}")
print(f"Result Output: {res.result}")
print()

# We can also look at the context for the chosen result and
# any other results that weren't chosen.
# (This prompt tends to take 2 attempts. If it only takes one, try re-running it.)
print(f"Total Generation Attempts: {len(res.sample_generations)}")
print()

print("Getting index of another result.")
index = 0  # Just choose the first one.

print(
    "If the below is the same output, try re-running this program to get multiple attempts."
)
print(f"Different attempted output: {res.sample_generations[index]}")
print()

# We can see the context that created this output.
gen_ctx = res.sample_contexts[index]
print(f"Previous step in generating this result was: {gen_ctx.previous_node.node_data}")  # type: ignore
print()

# We can also see what the validation context looked like.
req, val_result = res.sample_validations[index][0]
print(
    f"Getting context when evaluating the above output against Req({req.description})."
)
val_ctx = val_result.context

print(f"Output of the validation for this requirement: {val_ctx.node_data}")  # type: ignore
