# pytest: huggingface, e2e, slow
"""How `ALoraRequirement` routes validation through a fast adapter.

Uses the catalog-native `requirement-check` adapter (registered here via
`IntrinsicAdapter`, still the only working way to drive `_generate_from_intrinsic` —
see #1144) to compare aLoRA-backed validation against full LLM-as-judge generation
for the same requirement. `LLMaJRequirement` is used for the comparison because
`ALoraRequirement` always routes through the registered adapter regardless of
`backend.default_to_constraint_checking_alora`.

For loading a fully custom, non-catalog adapter with your own output schema
(not the generic `{"requirement_check": {"score": ...}}` shape), see
`stembolts_intrinsic.py` in this directory — it has no standalone entry point of
its own; `102_example.py` drives it interactively (reads from stdin, so it isn't
runnable as an automated example).
"""

import time

from mellea import MelleaSession, model_ids
from mellea.backends.adapters import AdapterType
from mellea.backends.adapters.adapter import IntrinsicAdapter
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import GenerateLog, ValidationResult
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import ALoraRequirement, LLMaJRequirement, Requirement

backend = LocalHFBackend(model_id=model_ids.IBM_GRANITE_4_1_3B, cache=SimpleLRUCache(5))

m = MelleaSession(backend=backend, ctx=ChatContext())

# Register the aLoRA variant of the catalog's requirement-check adapter. Without
# this, ALoraRequirement logs a warning and falls back to regular generation,
# whose prompt asks for a plain "yes"/"no" answer — but the result is still
# parsed as JSON by requirement_check_to_bool: a plain yes/no reply raises
# json.JSONDecodeError (it is not JSON), and a JSON reply that doesn't match
# the schema raises AdapterSchemaMismatchError — either way the error
# propagates out of validate() instead of returning a failed check. The ALORA
# type is also load-bearing here: routing only looks up ("alora",), so
# registering the LORA variant instead would hit the same failure.
backend.add_adapter(
    IntrinsicAdapter(  # emits a DeprecationWarning — see module docstring, #1144
        "requirement-check",
        adapter_type=AdapterType.ALORA,
        base_model_name=backend.base_model_name,
    )
)

description = "The summary must mention the suspected cause of failure."

# define a requirement
alora_check = ALoraRequirement(description)

res = m.instruct(
    "Write a triage summary based on this technician note: Oil seepage around "
    "piston rings suggests seal degradation.",
    strategy=None,
)

print("==== Generation =====")
print(f"Model Output: {res}")
print(
    f"Generation Prompt: {m.last_prompt()}"
)  # retrieve prompt information from session context


def validate_reqs(
    reqs: list[Requirement], label: str
) -> tuple[float, list[ValidationResult]]:
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

# Warm up both paths before timing. The *first* call against a freshly-registered
# aLoRA adapter pays a one-time PEFT weight-load cost that has nothing to do with
# per-call latency; the LLM-as-judge call has no such cost, but gets a warm-up too
# so both measurements below are on equal footing.
m.validate([alora_check])
m.validate([llmaj_check])

# ALoraRequirement always routes through the registered aLoRA adapter.
computetime_alora, _ = validate_reqs([alora_check], "aLoRA")

# LLMaJRequirement always bypasses adapters, regardless of what's registered —
# the only way to get a genuine no-adapter timing comparison for the same check.
computetime_llmaj, _ = validate_reqs([llmaj_check], "LLM-as-judge")

print(f"aLoRA validation:        {computetime_alora:.3f}s")
print(f"LLM-as-judge validation: {computetime_llmaj:.3f}s")
print(
    "NOTE: whichever way these numbers land, they are not measuring aLoRA's "
    "architectural advantage — reusing an already-computed KV cache instead of "
    "recomputing the context under adapter-modified weights. `generate_from_context` "
    "never routes through that KV-cache-reuse path (`_generate_from_context_with_kv_cache`; "
    "reachable today only by calling it directly, see `docs/kv_smash/hf_example.py`), so "
    "neither call above reuses anything. What actually separates them here is mostly the "
    "token-count difference between a JSON+score output and a one-word yes/no answer."
)
