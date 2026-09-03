---
title: "Tutorial: Adapter Schema Migrations"
description: "Pin an adapter function's Hugging Face revision and handle AdapterSchemaMismatchError so a breaking schema change doesn't take your program down."
# diataxis: tutorial
---

An adapter function's output schema is defined by its `io.yaml`, hosted
alongside its weights on Hugging Face Hub. If the maintainer changes that
schema — adding, renaming, or removing a field the adapter's `IOContract`
requires — any program still pointed at `main` (or another mutable branch)
picks up the new schema on its next download, with no warning. This
tutorial covers the two mechanisms that keep that from breaking your
program: revision pinning, and handling the exception a mismatch raises.

By the end you will have covered:

- Why `LocalFileBinding`/`IntrinsicsCatalogEntry` pin `revision` to a commit
  SHA rather than tracking `main`
- How to pin (or intentionally un-pin) a composed adapter's revision
- Catching `AdapterSchemaMismatchError` instead of letting it propagate
- Reading `ValidationResult.error` to distinguish "requirement not met" from
  "output unparsable" when using `ALoraRequirement`

**Prerequisites:** `pip install "mellea[hf]"`, a GPU or Apple Silicon Mac.

---

## Step 1: Understand what `revision` actually pins

`LocalFileBinding.revision` (and the built-in catalog's own
`IntrinsicsCatalogEntry.revision`) is a Hugging Face revision — a branch
name, tag, or commit SHA — passed straight through to the Hub download call.
Mellea's built-in catalog entries pin to commit SHAs by convention, precisely
so that a maintainer publishing a new `io.yaml` on `main` doesn't silently
change what your already-deployed program downloads:

```python
# Requires: mellea[hf]
from mellea.backends.adapters.catalog import fetch_intrinsic_metadata

metadata = fetch_intrinsic_metadata("requirement-check")
print(metadata.repo_id, metadata.revision)
# ibm-granite/granitelib-core-r1.0 d0a2a96a4cd07e96f0fe7ca29a42bfe088299d43
```

`LocalFileBinding.resolved_revision()` uses this pinned value whenever you
construct a binding with `revision=None` (the default for
`LocalFileBinding.from_catalog(name)`) — you get the reproducibility for
free for a catalog adapter. A **custom** adapter has no catalog entry, so
`revision=None` there raises instead (see
[Adding a custom adapter function](./07-custom-adapter-function.md)) —
pinning is not optional for a custom adapter, only automatic for a catalog one.

## Step 2: Pin explicitly when you need a specific version

If a maintainer ships a breaking `io.yaml` change on a new branch or tag
before promoting it to the pinned revision Mellea's catalog tracks, or if
you're testing against an upcoming version ahead of a Mellea release that
updates the pin, construct the binding with an explicit `revision` instead
of `from_catalog`:

```python
from mellea.backends.adapters import Adapter, Identity, LocalFileBinding, get_io_contract
from mellea.backends.adapters.catalog import AdapterType, fetch_intrinsic_metadata
from mellea.backends.huggingface import LocalHFBackend

metadata = fetch_intrinsic_metadata("requirement-check")
backend = LocalHFBackend(model_id="ibm-granite/granite-3.2-8b-instruct")

adapter = Adapter(
    identity=Identity(
        name="requirement-check",
        adapter_type="alora",
        capability=metadata.effective_capability,
    ),
    io_contract=get_io_contract("requirement-check"),
    weights=LocalFileBinding(
        name="requirement-check",
        adapter_type=AdapterType.ALORA,
        repo_id=metadata.repo_id,
        # Explicit revision — the "v2" candidate you're validating against,
        # not the catalog's currently-pinned "v1" SHA above.
        revision="some-future-commit-sha-or-branch-name",
    ),
)
backend.add_adapter(adapter)
```

Registering two revisions of the same adapter name on one backend at once
isn't supported — `add_adapter` keys the registry on
`f"{name}_{adapter_type}"`, so the second registration is refused. Use two
backend instances (or two processes) if you need to compare v1 and v2 output
side by side.

## Step 3: Handle a schema mismatch gracefully

Whether the mismatch comes from an unpinned adapter drifting underneath you,
or from deliberately testing a new revision, the failure mode is the same:
`IOContract.parse` raises `AdapterSchemaMismatchError` when the model's JSON
output doesn't satisfy the declared contract.

Calling an adapter directly through `Intrinsic`/`mfuncs.act()` propagates
this exception — catch it where you call:

```python
from mellea.backends.adapters import AdapterSchemaMismatchError
import mellea.stdlib.functional as mfuncs
from mellea.stdlib.components import Intrinsic

try:
    out, _ = mfuncs.act(Intrinsic("requirement-check", intrinsic_kwargs={
        "requirement": "The assistant is helpful."
    }), ctx, backend)
except AdapterSchemaMismatchError as e:
    print(f"Schema mismatch: expected {e.expected_keys}, observed {e.observed_keys}")
    if e.reason:
        print(f"Reason: {e.reason}")
```

`ALoraRequirement` handles this differently: it surfaces the error on
`ValidationResult.error` rather than raising, and fails the check closed
(`bool(result)` is `False`) so a schema mismatch is never silently treated
as "requirement not met":

```python
val_res = m.validate([requirement])
for result in val_res:
    if result.error is not None:
        print(f"Adapter output was unparsable: {result.error}")
    elif not bool(result):
        print("Requirement genuinely not met")
```

This distinction matters for monitoring: alert on `result.error is not None`
separately from your ordinary pass/fail rate — a spike in schema-mismatch
errors after an adapter update is a signal to re-pin or roll back, not
evidence your model got worse at the task.

**See also:**
[Adapter functions](../advanced/intrinsics.md) |
[Adding a custom adapter function in 20 lines](./07-custom-adapter-function.md) |
[Reading adapter function telemetry](./09-adapter-function-telemetry.md)
