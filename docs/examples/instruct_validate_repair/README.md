# Instruct-Validate-Repair Examples

Examples demonstrating Mellea's core instruct-validate-repair (IVR) paradigm for generating and constraining model outputs.

## Examples

| File | Description |
| :--- | :--- |
| [`101_email.py`](./101_email.py) | Basic email generation using `m.instruct()` without constraints. |
| [`101_email_with_requirements.py`](./101_email_with_requirements.py) | Constraining output using string-based requirements with automatic validation and repair. |
| [`101_email_with_validate.py`](./101_email_with_validate.py) | Explicit validation step using `m.validate()` to inspect requirement satisfaction. |
| [`101_email_comparison.py`](./101_email_comparison.py) | Comparing outputs generated with and without validation requirements. |
| [`advanced_email_with_validate_function.py`](./advanced_email_with_validate_function.py) | Custom validation functions using `simple_validate()` and `RejectionSamplingStrategy`. |
| [`multiturn_strategy_example.py`](./multiturn_strategy_example.py) | Conversational multi-turn repair using `MultiTurnStrategy` and `ChatContext`. |
| [`qiskit_code_validation/`](./qiskit_code_validation/) | Quantum computing code generation with external `flake8-qiskit-migration` linter validation. |
