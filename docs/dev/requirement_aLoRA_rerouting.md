#  Rerouting Requirement Actions in `Backend.generate_*` calls

Backend will often re-route a `generate` call where `action : Requirement` to an ALora. This document explains how and why that happens.

## The Requirement Rerouting Rule

## The Simple Rule

The simplest version of the Requirement Rerouting Rule is:

> The most specific constraint checking method will be used when validating generic `Requirement`s.

The actual rule is slightly more complicated.

## The Actual Rule

If a `Requirement` is validated using a backend that could either use a `requirement_check` aLoRA or perform an LLMaJ prompt on the underlying model, then the aLoRA is used for validation, even if the `backend.generate_from_context` method is called instead of the `backend._generate_from_intrinsic` method.

There are three exceptions to this rule:
1. `Backend.default_to_constraint_checking_alora` is set to `False` (this parameter defaults to `True`).
2. The `Requirement` has a more specific subtype that indicates a more specific intent (`LLMaJRequirement`). 
3. The `ALoRA` requirement checker throws an exception.

There is an exception (or disambiguation) to the first exception: If the user provides an `ALoRARequirement`, then the `backend.generate_from_context` call is rerouted to the constraint checking LoRA, regardless of the value of `default_to_constraint_checking_alora`.

## Decision Rationale

### Background and Problem Statement

The `stdlib` has a `Requirement` class whose `validate` behavior is an LLMaJ call.

Suppose that the user creates a backend and then adds a generic constraint checking aLoRA:

```python
from mellea import start_session
from mellea.core import Requirement
from mellea.backends.adapters import GraniteCommonAdapter

m = start_session(
    "huggingface.LocalHFBackend:ibm-granite/granite-3.2-8b-instruct")

# By default, the AloraRequirement uses a GraniteCommonAdapter with "requirement_check".
m.backend.add_adapter(GraniteCommonAdapter("ibm-granite/rag-intrinsics-lib", "requirement_check", base_model_name="granite-3.2-8b-instruct"))

m.instruct(
    "Corporate wants you to find the difference between these two strings:\n\naaa\naba")
assert m.validate(Requirement(
    description="The answer should mention that one of the strings has the letter b while the other doesn't."))
```

Both the underlying model and the aLoRA adapter know how to validate this requirement, so which should be used?

## Alternatives to the Proposed Rule

1. Avoid the problem by forcing the user to be more explicit.
2. Respect control flow in the backends/alora mixins, and have the MelleaSession or the user explicitly implement the appropriate control flow.
3. Have the `Requirement.validate` implementation specify whatever control flow is desired for that particular requirement.

### Advantages

1. Reduced cognitive load. To first approximation, there is a simple rule that produces unsurprising results. The exceptions are rare and require explicit intervention from the user. If these exceptions are used, the user almost certainly knows exactly what they are doing.
2. Control is retained. If the user wants to specify the precise semantics of their validate call, then they can use the mpore specific `LLMaJRequirement` and `ALoraRequirement` classes.
3. The backend is the one that needs to make the choice about whether to handle KV cache.


### Disadvantages

All backends that implement the aLoRA mixin need to implement this semantics. 

 * This might be a blessing in disguise. It's actually not clear that ALora context construction can be done WLOG outside of the specific backend.
 * That code is written rarely in any case.
 * Depending on the truth of the first bullet point's conjecture, we can mitigate by implementing this routing in `m.validate` so that even if a backend contributor gets this wrong the proper behavior is still usually observed by most users.
