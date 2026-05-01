# Safety Examples (Removed)

The `GuardianCheck` example files that previously lived here have been deleted.
`docs/examples/intrinsics/guardian_core.py`, `factuality_detection.py`,
`factuality_correction.py`, and `policy_guardrails.py` are the replacements.

## Migration gap: `RepairTemplateStrategy` + Guardian

The old `repair_with_guardian.py` demonstrated using `GuardianCheck` as a
`Requirement` inside `RepairTemplateStrategy` — the Guardian verdict (including
its chain-of-thought `_reason` string) was fed back into the repair loop as repair
guidance. This pattern **has no direct equivalent** in the Guardian Intrinsics API:

- Guardian Intrinsics return a `float` score, not a `Requirement` result, so they
  cannot be passed to `m.validate()` or used directly in `RepairTemplateStrategy`.
- The `thinking=True` / `_reason` chain-of-thought output from `GuardianCheck` is
  not exposed in the new API.

If you need repair-on-safety-failure behaviour with the new API, the closest
approach is to call `guardian.guardian_check()` manually after generation and
re-invoke `m.instruct()` with an additional requirement on failure.

## Related Documentation

- [Safety Guardrails (current)](../../docs/docs/how-to/safety-guardrails.md)
- [Security and Taint Tracking (deprecated)](../../docs/docs/advanced/security-and-taint-tracking.md)
