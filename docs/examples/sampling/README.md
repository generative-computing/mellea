# Sampling Examples

Pre-configured sampling presets for constrained generation tasks. A preset bundles
a set of requirements with a repair strategy so a common task works in one call —
as an alternative to assembling requirements individually. See
[`requirements/`](../requirements/README.md) for the lower-level building blocks
these presets wrap.

## Examples

| File | Description |
| :--- | :--- |
| [`python_code_generation_preset.py`](./python_code_generation_preset.py) | Python code generation preset bundling extraction, syntax, and import requirements with automatic repair feedback. By default (`use_sandbox=False`) generated code is parsed but not executed, and imports are checked only when `allowed_imports` is set. Pass `use_sandbox=True` to run it in Docker. |
| [`python_plotting_preset.py`](./python_plotting_preset.py) | Matplotlib plotting preset bundling headless backend configuration and savefig() output validation. Most examples in this file override the default to `use_sandbox=False` (static validation); `example_without_output_path` uses the sandbox default and requires Docker. |
