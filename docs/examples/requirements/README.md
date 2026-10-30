# Requirements Examples

Examples demonstrating Mellea's requirements for validating generated Python code,
including matplotlib plotting checks and execution of the generated code under an
execution tier. For choosing a tier and what a capability policy enforces, see
[Execution tiers](../../docs/how-to/tools-and-agents.md#execution-tiers). For the
requirements system itself, see
[Requirements system](../../docs/concepts/requirements-system.md) and
[Write custom verifiers](../../docs/how-to/write-custom-verifiers.md). For ready-made
bundles of these requirements, see [`sampling/`](../sampling/).

## Examples

| File | Description |
| :--- | :--- |
| [`matplotlib_plotting.py`](./matplotlib_plotting.py) | Demonstrates how to use matplotlib-specific requirements to validate code that generates plots. No model needed. |
| [`code_generation_and_execution.py`](./code_generation_and_execution.py) | Demonstrates the complete pipeline of code generation, data extraction, and graph visualization. |
| [`code_generation_and_execution_bash.py`](./code_generation_and_execution_bash.py) | Demonstrates Python code generation with `bash_executor` for CLI data processing tasks. |
