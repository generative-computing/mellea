# pytest: skip, huggingface, requires_heavy_ram, llm
# SKIP REASON: Example broken since intrinsics refactor - see issue #385

import time

from mellea import MelleaSession
from mellea.backends.adapters import AdapterType, GraniteCommonAdapter, catalog
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import GenerateLog
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import ALoraRequirement, Requirement

# Add the stembolts check to the intrinsics catalog.
# TODO-nrf this is hacky af.
catalog._INTRINSICS_CATALOG_ENTRIES.append(
    catalog.IntriniscsCatalogEntry(
        name="stembolts-checker", repo_id="nfulton/stembolts-checker"
    )
)
catalog._INTRINSICS_CATALOG = {e.name: e for e in catalog._INTRINSICS_CATALOG_ENTRIES}


# TODO-nrf also pretty freaking hacky.
class StemboltAdapter(GraniteCommonAdapter):
    # TODO how do I specify generation_prompt="<|start_of_role|>check_requirement<|end_of_role|>"???
    # TODO : just use Literal["alora", "lora"] instead of AdapterType.
    def __init__(self):
        super().__init__(
            intrinsic_name="stembolts-checker",
            base_model_name="granite-3.3-2b-instruct",
        )


# Define a backend
from mellea.backends.model_options import ModelOption

backend = LocalHFBackend(
    model_id="ibm-granite/granite-3.3-2b-instruct", cache=SimpleLRUCache(5)
)

# Add the adapter to the backend.
# TODO-nrf This is exactly the sort of thing I should be getting for free from all of these expensive abstractions...
backend.add_adapter(StemboltAdapter())

# Create M session
# TODO-nrf super weird flow here this whole thing above this line should be like 2 lines of code.
m = MelleaSession(backend, ctx=ChatContext())

# run instruction with requirement attached on the base model

# define a requirement
# TODO: we should be able to pass the adapter itself, or at the very least name should be a public property of Adapter.
failure_check = ALoraRequirement(
    "The failure mode should not be none.", intrinsic_name="stembolts-checker"
)

res = m.instruct(
    "Write triage summaries based on technician note 'Oil seepage around piston rings suggests seal degradation'",
    requirements=[failure_check],
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
        print(f"- [{val_res[i]}]: {r.description}")

    # Print prompts using the logs list
    print("Prompts:")
    for log in logs:
        if isinstance(log, GenerateLog):
            print(f" - {{prompt: {log.prompt}\n   raw result: {log.result.value} }}")  # type: ignore

    return end_time - start_time, val_res


# run with aLora -- which is the default if the constraint alora is added to a model
computetime_alora, alora_result = validate_reqs([failure_check])

# NOTE: This is not meant for use in regular programming using mellea, but just as an illustration for the speedup you can get with aloras.
# force to run without alora
backend.default_to_constraint_checking_alora = False
computetime_no_alora, no_alora_result = validate_reqs([failure_check])

print(
    f"Speed up time with using aloras is {(computetime_alora - computetime_no_alora) / computetime_no_alora * 100}% -- {computetime_alora - computetime_no_alora} seconds, not normalized for token count."
)
