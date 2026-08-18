# pytest: huggingface, e2e, qualitative, slow
"""How `ALoraRequirement` routes validation through a fast adapter.

Uses the catalog-native `requirement-check` adapter (registered here via
`IntrinsicAdapter`, still the only working way to drive `_generate_from_intrinsic` —
see #1144) to compare aLoRA-backed validation against full LLM-as-judge generation
for the same requirement. `LLMaJRequirement` is used for the comparison because
`ALoraRequirement` always routes through the registered adapter regardless of
`backend.default_to_constraint_checking_alora`.

For loading a fully custom, non-catalog adapter with your own output schema
(not the generic `{"requirement_check": {"score": ...}}` shape), see
`stembolts_intrinsic.py` and `102_example.py` in this directory.
"""

import time

from mellea import MelleaSession
from mellea.backends.adapters import AdapterType
from mellea.backends.adapters.adapter import IntrinsicAdapter
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import GenerateLog
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import ALoraRequirement, LLMaJRequirement, Requirement

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b", cache=SimpleLRUCache(5))

m = MelleaSession(backend=backend, ctx=ChatContext())

# Register the aLoRA variant of the catalog's requirement-check adapter. Without
# this, ALoraRequirement finds no matching adapter and silently falls back to
# regular generation (a warning is logged, but validation still "succeeds").
backend.add_adapter(
    IntrinsicAdapter(
        "requirement-check",
        adapter_type=AdapterType.ALORA,
        base_model_name=backend.base_model_name,
    )
)

description = "The summary must mention the suspected cause of failure."

# define a requirement
failure_check = ALoraRequirement(description)
failure_check.check_only = True

res = m.instruct(
    "Write a triage summary based on this technician note: Oil seepage around "
    "piston rings suggests seal degradation.",
    requirements=[failure_check],
    strategy=None,
)

print("==== Generation =====")
print(f"Model Output: {res}")
print(
    f"Generation Prompt: {m.last_prompt()}"
)  # retrieve prompt information from session context


def validate_reqs(reqs: list[Requirement], label: str):
    """Validate the requirements against the last output in the session."""
    print(f"==== Validation ({label}) =====")

    # helper to collect validation prompts (because validation calls never get added to session contexts).
    logs: list[GenerateLog] = []

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

    return delta_t, val_res


llmaj_check = LLMaJRequirement(description)

# Warm up both paths first: the *first* call against a freshly-registered aLoRA
# adapter also pays a one-time PEFT weight-load cost that has nothing to do with
# per-call latency, so timing a cold call would overstate the difference below.
m.validate([failure_check])
m.validate([llmaj_check])

# ALoraRequirement always routes through the registered aLoRA adapter.
computetime_alora, alora_result = validate_reqs([failure_check], "aLoRA")

# LLMaJRequirement always bypasses adapters, regardless of what's registered --
# the only way to get a genuine no-adapter timing comparison for the same check.
computetime_llmaj, llmaj_result = validate_reqs([llmaj_check], "LLM-as-judge")

print(f"aLoRA validation:        {computetime_alora:.3f}s")
print(f"LLM-as-judge validation: {computetime_llmaj:.3f}s")
print(
    "NOTE: these numbers do not demonstrate a speedup either way, and that's "
    "expected. aLoRA's architectural advantage is reusing an already-computed "
    "KV cache instead of recomputing the context under adapter-modified weights -- "
    "mellea's adapter-activation path doesn't exercise that yet (KV-cache reuse is "
    "experimental and gated behind `cache=True` context blocks, not wired into "
    "intrinsics). Over one short call, the token-count difference between a "
    "JSON+score output and a one-word yes/no answer dominates instead."
)
