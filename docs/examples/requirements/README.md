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
| [`matplotlib_plotting.py`](./matplotlib_plotting.py) | Calling `MatplotlibHeadlessBackend`, `PlotFileSaved` and `PlotDependenciesAvailable` directly on sample code, with passing and failing cases. No model needed. |
| [`code_generation_and_execution.py`](./code_generation_and_execution.py) | Natural-language request to saved graph: CSV loading, code generation, execution via `PythonExecutionReq(execution_tier="local")`, and plot validation. Takes `--csv`, `--interactive` and `--output`. |
| [`code_generation_and_execution_bash.py`](./code_generation_and_execution_bash.py) | Generating Python that orchestrates `bash_executor()` calls for CLI data processing, validated with `PythonSyntaxValid` and run with `python_tool`. Takes `--workspace` and `--interactive`. |
