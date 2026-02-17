# pytest: skip, huggingface, requires_heavy_ram, llm
# SKIP REASON: Example broken since intrinsics refactor - see issue #385

import time

from mellea import MelleaSession
from mellea.backends.adapters.adapter import CustomGraniteCommonAdapter
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import GenerateLog
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import ALoraRequirement, Requirement

backend = LocalHFBackend(
    model_id="ibm-granite/granite-3.3-2b-instruct", cache=SimpleLRUCache(5)
)

m = MelleaSession(backend=backend, ctx=ChatContext())


class StemboltAdapter(CustomGraniteCommonAdapter):
    def __init__(self):
        super().__init__(
            model_id="nfulton/stembolts",
            intrinsic_name="stembolts",
            base_model_name="granite-3.3-2b-instruct",
        )


granite_33_2b_stembolt_adapter = StemboltAdapter()

backend.add_adapter(granite_33_2b_stembolt_adapter)

# define a requirement
failure_check = ALoraRequirement(
    "The diagnostic confidence should be in the unit interval and greater than 0.9.",
    intrinsic_name=granite_33_2b_stembolt_adapter.intrinsic_name,
)
failure_check.check_only = True

res = m.instruct(
    "Oil seepage around piston rings suggests seal degradation",
    requirements=[failure_check],
    strategy=None,
)

print("==== Generation =====")
print(f"Model Output: {res}")
print(
    f"Generation Prompt: {m.last_prompt()}"
)  # retrieve prompt information from session context


def validate_reqs(reqs: list[Requirement]):
    """Validate the requirements against the last output in the session."""
    print("==== Validation =====")
    print(
        "using aLora"
        if backend.default_to_constraint_checking_alora
        else "using NO alora"
    )

    # helper to collect validation prompts (because validation calls never get added to session contexts).
    logs: list[GenerateLog] = []  # type: ignore

    # Run the validation. No output needed, because the last output in "m" will be used. Timing added.
    start_time = time.time()
    val_res = m.validate(reqs, generate_logs=logs)
    end_time = time.time()
    delta_t = end_time - start_time

    print(f"Validation took {delta_t} seconds.")
    print("Validation Results:")

    # Print list of requirements and validation results
    for i, r in enumerate(reqs):
        print(f"- {r.description}: [{val_res[i].reason}]")

    # Print prompts using the logs list
    print("Prompts:")
    for log in logs:
        if isinstance(log, GenerateLog):
            print(f" - {{prompt: {log.prompt}\n   raw result: {log.result.value} }}")  # type: ignore

    return end_time - start_time, val_res


# NOTE: This is not meant for use in regular programming using mellea, but just as an illustration for the speedup you can get with aloras.
# force to run without alora
backend.default_to_constraint_checking_alora = False
computetime_no_alora, no_alora_result = validate_reqs([failure_check])

# run with aLora -- which is the default if the constraint alora is added to a model
backend.default_to_constraint_checking_alora = True
computetime_alora, alora_result = validate_reqs([failure_check])


print(
    f"Speed up time with using aloras is {((computetime_alora - computetime_no_alora) / computetime_no_alora * 100):.2f}% ({computetime_alora - computetime_no_alora} seconds). This speedup is absolute -- not normalized for token count."
)
