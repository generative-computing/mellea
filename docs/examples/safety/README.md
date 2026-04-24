# Safety Examples (Deprecated)

> **Deprecated.** These examples use the `GuardianCheck` API, which is deprecated
> as of Mellea v0.4 and will emit `DeprecationWarning` on use.
>
> For the current Guardian API, see:
> - **[`../intrinsics/`](../intrinsics/)** — replacement examples using Guardian Intrinsics
> - **[Safety Guardrails how-to](https://mellea.dev/how-to/safety-guardrails)** — full documentation

## Files

### guardian.py

`GuardianCheck` examples using Granite Guardian 3.3 8B via Ollama.

### guardian_huggingface.py

`GuardianCheck` examples using a HuggingFace backend with shared backend reuse.

### repair_with_guardian.py

`GuardianCheck` combined with `RepairTemplateStrategy`. Note: this pattern has no direct
equivalent in the Guardian Intrinsics API.

## Related Documentation

- [Security and Taint Tracking (deprecated)](../../../docs/docs/advanced/security-and-taint-tracking.md)
- [Safety Guardrails (current)](../../../docs/docs/how-to/safety-guardrails.md)
