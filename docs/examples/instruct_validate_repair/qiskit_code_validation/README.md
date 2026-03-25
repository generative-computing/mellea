# Qiskit Code Validation with Instruct-Validate-Repair

This example demonstrates using Mellea's Instruct-Validate-Repair (IVR) pattern to generate Qiskit quantum computing code that automatically passes `flake8-qiskit-migration` validation rules (QKT rules).

## What This Example Does

Takes a prompt containing deprecated Qiskit code and:
1. Detects QKT violations in the input code
2. Passes those violations to the LLM as context
3. Generates corrected code that passes QKT validation
4. Automatically repairs the code if validation fails (up to 5 attempts)

## Quick Start

```bash
# Run the example (uses default deprecated code prompt)
uv run docs/examples/instruct_validate_repair/qiskit_code_validation/qiskit_code_validation.py
```

Dependencies (`mellea`, `flake8-qiskit-migration`) are automatically installed.

## Requirements

- **Ollama backend** running locally (`ollama serve`)
- **Compatible model**: e.g., `hf.co/Qiskit/mistral-small-3.2-24b-qiskit-GGUF:latest` or `granite4:small-h`
- **flake8-qiskit-migration**: Automatically installed when using `uv run`

## How It Works

### The IVR Pipeline

1. **Pre-condition validation**: Validates the input prompt and any code it contains
2. **Instruction**: LLM generates code following structured requirements
3. **Post-condition validation**: Validates generated code against QKT rules (see [Qiskit Migration Guide](https://docs.quantum.ibm.com/api/migration-guides))
4. **Repair loop**: Automatically repairs code that fails validation (up to 5 attempts)

### Sampling Strategies

The example supports two repair strategies (see [Sampling Strategies](../README.md#sampling-strategies)):

- **RepairTemplateStrategy** (default): Adds validation failure reasons directly to the instruction and retries generation
- **MultiTurnStrategy**: Builds conversation history by adding validation failures as new user messages

To switch strategies, edit the `use_multiturn_strategy` variable in `test_qiskit_code_validation()`

**Note**: `MultiTurnStrategy` requires `ChatContext()` while `RepairTemplateStrategy` works with `SimpleContext()`. The example automatically selects the appropriate context based on your strategy choice.

#### Strategy Performance Comparison

Benchmarks on `mistral-small-3.2-24b-qiskit` model (pass rates measure QKT validation only, not correctness):

| Dataset | Strategy | First Pass | Post-Repair |
|---------|----------|------------|-------------|
| **QHE** | RepairTemplate | 78.2% | **99.3%** |
|         | MultiTurn | 77.5% | 96.7% |
| **QKT** | RepairTemplate | 54.1% | **83.8%** |
|         | MultiTurn | 37.8% | 70.3% |

**Datasets:**
- **QHE** (QiskitHumanEval): Code generation tasks testing general Qiskit programming
- **QKT**: Qiskit version migration tasks requiring fixes to deprecated APIs

**Note:** These benchmarks measure whether generated code passes QKT validation rules, not whether the code correctly solves the prompt. Both aspects are important for production use.

### Code Structure

```
qiskit_code_validation/
├── qiskit_code_validation.py   # Main example
├── validation_helpers.py       # Validation utilities
└── README.md                   # This file
```

**validation_helpers.py** provides:
- `extract_code_from_markdown()`: Extracts code from markdown blocks
- `validate_qiskit_migration()`: Validates against QKT rules
- `validate_input_code()`: Pre-validates input prompts

## Trying Different Prompts

To try different prompts, edit the `prompt` variable in `test_qiskit_code_validation()` function. Here are some examples you can copy/paste:

### Simple Prompts

**Bell State Circuit:**
```python
prompt = "create a bell state circuit"
```

**List Backends:**
```python
prompt = "use qiskit to list fake backends"
```

**Random Circuit:**
```python
prompt = "give me a random qiskit circuit"
```

### Code Completion Prompts

**Toffoli Gate:**
````python
prompt = """Complete this code:
```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.toffoli(0, 1, 2)

# draw the circuit
```
"""
````

**Entanglement Circuit:**
```python
prompt = """from qiskit import QuantumCircuit

# create an entanglement state circuit
"""
```

### Deprecated Code (Default)

The default prompt demonstrates fixing deprecated Qiskit APIs:

```python
prompt = """from qiskit import BasicAer, QuantumCircuit, execute

backend = BasicAer.get_backend('qasm_simulator')

qc = QuantumCircuit(5, 5)
qc.h(0)
qc.cnot(0, range(1, 5))
qc.measure_all()

# run circuit on the simulator"""
```

This code uses deprecated APIs (`BasicAer`, `execute`) that the LLM will automatically fix to use modern Qiskit APIs.

### Complex Prompts

**Runtime Service with Estimator:**
```python
prompt = """from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Estimator, Options, QiskitRuntimeService, Session

# create a Qiskit random circuit named "circuit" with 2 qubits, depth 2, seed 1.
# After that, generate an observable type SparsePauliOp("IY"). Run it in the backend "ibm_sherbrooke" using QiskitRuntimeService inside a session
# Instantiate the runtime Estimator primitive using the session and the options optimization level 3 and resilience level 2. Run the estimator
# Conclude the code printing the observable, expectation value and the metadata of the job."""
```

**Bell Circuit with Runtime Service:**
```python
prompt = """from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService

# define a Bell circuit and run it in ibm_salamanca using QiskitRuntimeService"""
```

## Expected Output

When you run the example with the default deprecated code prompt, you'll see:

````
====== Prompt ======
from qiskit import BasicAer, QuantumCircuit, execute

backend = BasicAer.get_backend('qasm_simulator')

qc = QuantumCircuit(5, 5)
qc.h(0)
qc.cnot(0, range(1, 5))
qc.measure_all()

# run circuit on the simulator
======================

Validation failed with 1 error(s):
QKT101: QuantumCircuit.cnot() has been removed in Qiskit 1.0; use `.cx()` instead

====== Result (83.5s) ======
```python
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit

backend = AerSimulator()

qc = QuantumCircuit(5, 5)
qc.h(0)
qc.cx(0, range(1, 5))  # Fixed: use .cx() instead of .cnot()
qc.measure_all()

job = backend.run(qc)
result = job.result()
```
I fixed the code by replacing `QuantumCircuit.cnot()` with `QuantumCircuit.cx()` as required by Qiskit 1.0. I also replaced the deprecated `BasicAer.get_backend('qasm_simulator')` with `AerSimulator()`. This code should now pass Qiskit migration validation (QKT rules).
======================

✓ Code passes Qiskit migration validation
````

**Note**: The exact output may vary depending on the model and its interpretation of the prompt.

## Changing the Model

To try a different model, edit the `model_id` variable in the `test_qiskit_code_validation()` function:

```python
# Uncomment one to try different models
# model_id = "granite4:micro-h"
# model_id = "granite4:small-h"
model_id = "hf.co/Qiskit/mistral-small-3.2-24b-qiskit-GGUF:latest"
```

**Note**: Smaller models (like `granite4:micro-h`) may not have enough Qiskit knowledge to pass validation consistently. The Qiskit-specific model or `granite4:small-h` work best.

## Troubleshooting

### Ollama Connection Refused
```
Error: Connection refused
```
**Solution**: Start Ollama with `ollama serve`

### Model Not Found
```
Error: model 'hf.co/Qiskit/mistral-small-3.2-24b-qiskit-GGUF:latest' not found
```
**Solution**: Pull the model first:
```bash
ollama pull hf.co/Qiskit/mistral-small-3.2-24b-qiskit-GGUF:latest
```

### Validation Always Fails
If using smaller models (e.g., `granite4:micro-h`), they may not have enough Qiskit knowledge. Try:
- Using a larger model (`granite4:small-h` or the Qiskit-specific model)
- Reducing prompt complexity
- Using simpler prompts

### Import Error: flake8-qiskit-migration
```
ModuleNotFoundError: No module named 'flake8_qiskit_migration'
```
**Solution**: Use `uv run` which auto-installs dependencies

## Future Work

The following enhancements are planned for future iterations:

1. **Enable Smaller Models** - Add system prompt or grounding context with Qiskit API documentation to help smaller models perform accurate migrations. This would allow removing the `pytest.mark.skip` marker and make the example run in standard test suites.
