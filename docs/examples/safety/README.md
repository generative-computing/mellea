# Safety examples

Guardian adapter functions used as requirement checks.

- [`repair_with_guardian.py`](repair_with_guardian.py): wraps
  `guardian_check()` in a `Requirement`, so a failed check feeds back into
  `RepairTemplateStrategy`.

The session needs a `ChatContext`: a Guardian check evaluates a conversation,
and the default `SimpleContext` keeps no history.

Prerequisites: `uv sync --extra hf`, plus a GPU or Apple Silicon Mac. See
[Safety Guardrails](../../docs/how-to/safety-guardrails.md) for the adapter
functions, and [`intrinsics/guardian_core.py`](../intrinsics/guardian_core.py)
for direct `guardian_check()` usage.
