# Qiskit Code Validation with Instruct-Validate-Repair

This example demonstrates using Mellea's Instruct-Validate-Repair (IVR) pattern to generate Qiskit quantum computing code that automatically passes `flake8-qiskit-migration` validation rules (QKT rules).

## What This Example Does

Takes a prompt containing deprecated Qiskit code and:
1. Detects QKT violations in the input code
2. Passes those violations to the LLM as context
3. Generates corrected code that passes QKT validation
4. Automatically repairs the code if validation fails (up to 10 attempts)

## Quick Start

```bash
# Run the example (uses default deprecated code prompt)
uv run docs/examples/instruct_validate_repair/qiskit_code_validation/qiskit_code_validation.py
```

Dependencies (`mellea`, `flake8-qiskit-migration`) are automatically installed.

## Requirements

- **Ollama backend** running locally (`ollama serve`)
- **Compatible model**: `hf.co/Qiskit/mistral-small-3.2-24b-qiskit-GGUF:latest` (recommended — domain-specialized; see [Changing the Model](#changing-the-model))
- **flake8-qiskit-migration**: Automatically installed when using `uv run`

## How It Works

### The IVR Pipeline

1. **Pre-condition validation**: Validates the input prompt and any code it contains
2. **Instruction**: LLM generates code following structured requirements
3. **Post-condition validation**: Validates generated code against QKT rules (see [Qiskit Migration Guide](https://docs.quantum.ibm.com/api/migration-guides))
4. **Repair loop**: Automatically repairs code that fails validation (up to 10 attempts)

### Sampling Strategies

The example supports two repair strategies (see [Sampling Strategies](../README.md#sampling-strategies)):

- **RepairTemplateStrategy** (default): Adds validation failure reasons directly to the instruction and retries generation
- **MultiTurnStrategy**: Builds conversation history by adding validation failures as new user messages

To switch strategies, edit the `use_multiturn_strategy` variable in `test_qiskit_code_validation()`

**Note**: `MultiTurnStrategy` requires `ChatContext()` while `RepairTemplateStrategy` works with `SimpleContext()`. The example automatically selects the appropriate context based on your strategy choice.

#### Strategy Performance Comparison

Benchmarks on `mistral-small-3.2-24b-qiskit` model, no system prompt:

| Dataset | Strategy | First Pass (QKT) | Post-Repair (QKT) |
|---------|----------|------------|-------------|
| **QHE** | RepairTemplate | 98.0% | **100%** |
|         | MultiTurn | **100%** | **100%** |
| **QKT** | RepairTemplate | 98.0% | **100%** |
|         | MultiTurn | 93.3% | **100%** |

**Datasets:**
- **QHE** (QiskitHumanEval): 151 general Qiskit code generation tasks
- **QKT**: 45 Qiskit version migration tasks requiring fixes to deprecated APIs

**Note:** Pass rates measure whether generated code passes QKT validation rules, not whether the code correctly solves the prompt. On QHE, the model achieves ~32.5% correctness when running the QHE check() test suite against the generated code. Full benchmark data and analysis are available in @ajbozarth's [toolbox repo](https://github.com/ajbozarth/toolbox/tree/main/mellea/qiskit_code_validation/benchmarking).

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

====== Result (23.1s, 2 attempt(s)) ======
```python
from qiskit_aer import AerSimulator, QuantumCircuit

backend = AerSimulator()

qc = QuantumCircuit(5, 5)
qc.h(0)
qc.cx(0, range(1, 5))
qc.measure_all()
```
======================

✓ Code passes Qiskit migration validation
````

**Note**: The exact output may vary depending on the model and its interpretation of the prompt.

## Changing the Model

To try a different model, edit the `model_id` variable in the `test_qiskit_code_validation()` function:

```python
model_id = "hf.co/Qiskit/mistral-small-3.2-24b-qiskit-GGUF:latest"
```

The default model is a Qiskit-specialized fine-tune of Mistral Small. It requires a large initial download (~15GB) but produces reliable results without a system prompt.

General-purpose models (e.g. `granite4:micro-h`) can be used as a lighter alternative but have significantly lower correctness on Qiskit tasks. When using a non-specialized model, set `system_prompt = QISKIT_SYSTEM_PROMPT` to improve results.

## Using Grounding Context

The `grounding_context` parameter accepts a `dict[str, str]` of additional context passed to the LLM alongside the prompt. Keys act as section labels and values are the content. This is useful for injecting relevant documentation snippets, RAG results, or API references at inference time.

**Example — injecting migration guide excerpts:**

```python
grounding_context = {
    "primitives_migration": (
        "SamplerV2 replaces the legacy execute() function. "
        "Use: sampler = SamplerV2(backend); job = sampler.run([circuit]); result = job.result()"
    ),
    "transpilation": (
        "Use generate_preset_pass_manager() instead of transpile(). "
        "Example: pm = generate_preset_pass_manager(optimization_level=1, backend=backend); isa_circuit = pm.run(circuit)"
    ),
}

code, success, attempts = generate_validated_qiskit_code(
    m, prompt, strategy, grounding_context=grounding_context
)
```

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
If using a general-purpose model, it may not have enough Qiskit knowledge to pass validation consistently. Try:
- Switching to the Qiskit-specialized model (`hf.co/Qiskit/mistral-small-3.2-24b-qiskit-GGUF:latest`)
- Setting `system_prompt = QISKIT_SYSTEM_PROMPT` to guide the model toward modern Qiskit APIs
- Using simpler prompts

### Import Error: flake8-qiskit-migration
```
ModuleNotFoundError: No module named 'flake8_qiskit_migration'
```
**Solution**: Use `uv run` which auto-installs dependencies

