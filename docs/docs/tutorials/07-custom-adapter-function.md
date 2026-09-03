---
title: "Tutorial: Adding a Custom Adapter Function"
description: "Compose your own trained LoRA/aLoRA adapter into Mellea without a shim class, in about 20 lines."
# diataxis: tutorial
---

This tutorial walks through registering a custom, non-catalog adapter
function by composing an `Adapter` directly — the replacement for the
deprecated `CustomIntrinsicAdapter` shim (Epic #929, issue #1144).

By the end you will have covered:

- Composing an `Adapter` from an `Identity`, an output contract, and a
  `LocalFileBinding`
- Why a custom adapter needs an explicit `revision`
- Registering it with a backend and invoking it through `Intrinsic`
- Validating output with `ALoraRequirement` against a custom adapter name

**Prerequisites:** `pip install "mellea[hf]"`, a trained LoRA/aLoRA adapter
uploaded to Hugging Face Hub (see
[LoRA and aLoRA adapters](../advanced/lora-and-alora-adapters.md) for
training and uploading), and a GPU or Apple Silicon Mac.

---

## Step 1: Compose the adapter

Mellea's built-in adapter functions (`answerability`, `requirement-check`,
etc.) are looked up from a catalog that supplies their `repo_id` and
`revision`. A custom adapter has no catalog entry, so you supply those
fields yourself:

```python
# Requires: mellea[hf]
from mellea.backends.adapters import Adapter, Identity, LocalFileBinding, get_io_contract
from mellea.backends.adapters.catalog import AdapterType

def custom_failure_check_adapter() -> Adapter:
    return Adapter(
        identity=Identity(name="custom-failure-check", adapter_type="alora"),
        # get_io_contract falls back to a permissive dict contract for names
        # outside the built-in catalog — sufficient here, since the real
        # output shape is enforced by the adapter's own io.yaml, not by
        # IOContract (which doesn't yet drive request/response handling).
        io_contract=get_io_contract("custom-failure-check"),
        weights=LocalFileBinding(
            name="custom-failure-check",
            adapter_type=AdapterType.ALORA,
            repo_id="your-org/my-adapter",  # your Hugging Face repo
            # A LocalFileBinding with revision=None resolves it from the
            # catalog — which has no entry for a custom name. Pin it
            # explicitly (a branch, tag, or commit SHA).
            revision="main",
        ),
    )
```

> **Trap:** Omitting `revision` here raises `ValueError: Unknown intrinsic
> name '...'` from deep inside `LocalFileBinding.resolved_revision()` — easy
> to misread as a registration failure rather than what it actually is: no
> catalog entry to fall back to.

## Step 2: Register it with a backend

```python
from mellea.backends.huggingface import LocalHFBackend

backend = LocalHFBackend(model_id="ibm-granite/granite-3.2-8b-instruct")
backend.add_adapter(custom_failure_check_adapter())  # downloads and loads the weights
```

`add_adapter` refuses a second registration under the same qualified name, so
guard repeated calls (e.g. inside a helper function) the same way the
built-in adapters do:

```python
qualified_name = "custom-failure-check_alora"
if qualified_name not in backend.list_adapters():
    backend.add_adapter(custom_failure_check_adapter())
```

## Step 3: Invoke it

There are two ways to invoke a registered adapter, depending on whether you
need raw output or a pass/fail validation.

### Directly, for raw output

```python
from mellea.backends.adapters.catalog import AdapterType
from mellea.stdlib.components.intrinsic import Intrinsic
import mellea.stdlib.functional as mfuncs
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext

# adapter_types is required for a name outside the catalog — Intrinsic has
# no catalog entry to look up adapter_types from otherwise.
action = Intrinsic("custom-failure-check", adapter_types=(AdapterType.ALORA,))
ctx = ChatContext().add(Message("user", "Observed black soot on intake."))
out, _ = mfuncs.act(action, ctx, backend)
print(out)
```

### As a validator, with `ALoraRequirement`

If your adapter's `io.yaml` transforms its output into
`{"requirement_check": {"score": <float>}}`, use it as a drop-in requirement
validator instead:

```python
from mellea import MelleaSession
from mellea.stdlib.requirements import ALoraRequirement

m = MelleaSession(backend, ctx=ChatContext())

failure_check = ALoraRequirement(
    "The failure mode must not be 'no_failure'.",
    intrinsic_name="custom-failure-check",
    # Required for the same reason as Intrinsic's adapter_types above.
    adapter_types=(AdapterType.ALORA,),
)
result = m.instruct(
    "Summarize this technician note: {{note}}",
    user_variables={"note": "High vibration at 3100 RPM."},
    requirements=[failure_check],
)
```

## What replaced the old approach

The deprecated `CustomIntrinsicAdapter` shim did two things under the hood
that this tutorial's composed `Adapter` replaces without a class of its own:

1. **Adapter registration** — `LocalFileBinding` already accepts an arbitrary
   `repo_id`, so no shim was ever needed for this half.
2. **Making the name usable by `Intrinsic`/`ALoraRequirement`** —
   `CustomIntrinsicAdapter` monkey-patched Mellea's global intrinsics catalog
   to add the custom name before constructing anything. `Intrinsic` and
   `ALoraRequirement` now accept `adapter_types` explicitly instead, so no
   catalog mutation is needed — only the caller's own process is affected,
   and there is no shared global state to leak between callers.

**See also:** [Adapter functions](../advanced/intrinsics.md) |
[LoRA and aLoRA adapters](../advanced/lora-and-alora-adapters.md) |
[Reading adapter function telemetry](./adapter-function-telemetry)
