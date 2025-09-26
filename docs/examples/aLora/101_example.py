import time

from mellea import MelleaSession
from mellea.backends.aloras.huggingface.granite_aloras import HFConstraintAlora
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.base import ChatContext, GenerateLog
from mellea.stdlib.requirement import ALoraRequirement, Requirement

# Define a backend and add the constraint aLora
backend = LocalHFBackend(
    model_id="ibm-granite/granite-3.2-8b-instruct", cache=SimpleLRUCache(5)
)

custom_stembolt_failure_constraint = HFConstraintAlora(
    name="custom_stembolt_failure_constraint",
    path_or_model_id="docs/examples/aLora/checkpoints/alora_adapter",  # can also be the checkpoint path
    generation_prompt="<|start_of_role|>check_requirement<|end_of_role|>",
    backend=backend,
)

backend.add_alora(custom_stembolt_failure_constraint)

# Create M session
m = MelleaSession(backend, ctx=ChatContext())

# define a requirement
failure_check = ALoraRequirement(
    "The failure mode should not be none.", alora=custom_stembolt_failure_constraint
)

# run instruction with requirement attached on the base model
res = m.instruct(
    """Write triage summaries based on technician note.
    1. Oil seepage around piston rings suggests seal degradation
    """,
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

print(f"Speed up time with using aloras is {computetime_alora - computetime_no_alora}")
