# Qiskit Code Validation with Instruct-Validate-Repair

This example demonstrates using Mellea's Instruct-Validate-Repair (IVR) pattern to generate Qiskit quantum computing code that automatically passes `flake8-qiskit-migration` validation rules (QKT rules), and, optionally, the problem's own test suite and the [LintQ](https://github.com/sola-st/LintQ) static analyzer for quantum programs.

## What This Example Does

Takes a prompt containing deprecated Qiskit code and:
1. Generates corrected code using the LLM
2. Validates the output against QKT rules, the test suite and LintQ
3. Automatically repairs the code if validation fails (up to 10 attempts)

`validation_helpers.py` also provides test-suite and LintQ validators for benchmark problems (Qiskit HumanEval, QuantumKatas); see [Validating with Test Suites and LintQ](#validating-with-test-suites-and-lintq).

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

1. **Instruction**: LLM generates code following structured requirements
2. **Post-condition validation**: Validates generated code against QKT rules (see [Qiskit Migration Guide](https://docs.quantum.ibm.com/api/migration-guides))
3. **Repair loop**: Automatically repairs code that fails validation (up to 10 attempts)

### Sampling Strategies

The example supports two repair strategies (see [Sampling Strategies](../README.md#sampling-strategies)):

- **RepairTemplateStrategy** (default): Adds validation failure reasons directly to the instruction and retries generation
- **MultiTurnStrategy**: Builds conversation history by adding validation failures as new user messages

To switch strategies, edit the `use_multiturn_strategy` variable in `test_qiskit_code_validation()`

**Note**: `MultiTurnStrategy` requires `ChatContext()` while `RepairTemplateStrategy` works with `SimpleContext()`. The example automatically selects the appropriate context based on your strategy choice.

#### Strategy Performance Comparison

Benchmarks on `mistral-small-3.2-24b-qiskit` model:

| Dataset | Strategy | First Pass (QKT) | Post-Repair (QKT) |
|---------|----------|------------|-------------|
| **QHE** | RepairTemplate | 97.4% | **100%** |
|         | MultiTurn | 95.4% | **100%** |
| **QKT** | RepairTemplate | 88.9% | **100%** |
|         | MultiTurn | **97.8%** | **100%** |

**Datasets:**
- **QHE** (QiskitHumanEval): 151 general Qiskit code generation tasks
- **QKT**: 45 Qiskit version migration tasks requiring fixes to deprecated APIs

**Note:** Pass rates measure whether generated code passes QKT validation rules, not whether the code correctly solves the prompt. On QHE, the model achieves ~27.8% correctness when running the QHE check() test suite against the generated code. Full benchmark data and analysis are available in @ajbozarth's [toolbox repo](https://github.com/ajbozarth/toolbox/tree/main/mellea/qiskit_code_validation/benchmarking).

### Code Structure

```
qiskit_code_validation/
├── qiskit_code_validation.py   # Main example (single prompt)
├── validation_helpers.py       # Validation utilities
└── README.md                   # This file
```

**validation_helpers.py** provides:
- `extract_code_from_markdown()`: Extracts code from markdown blocks
- `validate_qiskit_migration()`: Validates against QKT rules
- `validate_input_code()`: Pre-validates input prompts
- `validate_correctness()`: Runs code against a benchmark problem's own test suite
- `validate_lintq()`: Runs the LintQ CodeQL queries over the code

All validators return `(is_valid, error_message)`, so the message is fed straight back to the model as the repair reason.

### Adding LintQ to the Single-Prompt Example

Set `use_lintq = True` in `test_qiskit_code_validation()` to also reject code flagged by LintQ. This needs the [LintQ setup](#lintq-setup) described below.

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

## Validating with Test Suites and LintQ

Passing the QKT rules only shows that the code uses current Qiskit APIs, not that it solves the task or that it is free of quantum-specific bugs. For benchmark problems, `validate_correctness()` and `validate_lintq()` add two more checks:

| Validator | What it checks | Repair reason fed back to the model |
|-----------|----------------|-------------------------------------|
| **QKT rules** | `flake8-qiskit-migration`, as in the single-prompt example | The QKT rule messages |
| **Test suite** | Executes the code against the problem's own tests: `check(candidate)` for Qiskit HumanEval, `test_<entry_point>()` for QuantumKatas (30s timeout) | The assertion message or full traceback |
| **LintQ** | Runs the [LintQ](https://github.com/sola-st/LintQ) CodeQL queries (double measurement, operations after measurement, unused qubits, ...) | `[rule-id] message` per finding |

Both plug into the same `generate_validated_qiskit_code()` loop through its `extra_requirements` parameter. Running LintQ only once the tests pass avoids linting code that does not run, and saves time: each LintQ run builds a CodeQL database (around 20s).

```python
from validation_helpers import (
    extract_code_from_markdown,
    validate_correctness,
    validate_lintq,
)

from mellea.stdlib.requirements import req, simple_validate


def tests_and_lintq(problem: dict):
    """Pass the problem's tests, then LintQ."""

    def check(output: str) -> tuple[bool, str]:
        code = extract_code_from_markdown(output)
        passed, reason = validate_correctness(problem, code)
        return validate_lintq(code) if passed else (passed, reason)

    return req(
        "The code must pass the problem's test suite and raise no LintQ warnings",
        validation_fn=simple_validate(check),
    )


code, success, attempts = generate_validated_qiskit_code(
    m, problem["prompt"], strategy, extra_requirements=[tests_and_lintq(problem)]
)
```

> ⚠️ **Security**: the test-suite validator runs `exec()` on model-generated code and on the dataset's test code, with a timeout but **not** in a sandbox. Run it only on datasets you trust, on a machine where running arbitrary Python is acceptable.

### Setup

The benchmark dependencies, the datasets and LintQ are **not** part of this repository. Install and download them anywhere you like; the paths below are only examples.

#### Python Dependencies

The benchmark tests import Qiskit and a few plotting and graph libraries. Install them in the project environment:

```bash
uv pip install flake8-qiskit-migration "qiskit>=2.1.0,<2.5.0" qiskit-aer qiskit-ibm-runtime qiskit-ibm-transpiler matplotlib networkx pylatexenc seaborn
```

#### Datasets

The benchmarks are published by Qiskit on Hugging Face, with the same fields (`task_id`, `prompt`, `canonical_solution`, `test`, `entry_point`); each row is one `problem` for `validate_correctness()`.

| Benchmark | Tasks | Prompt style | License |
|-----------|-------|--------------|---------|
| [Qiskit HumanEval hard](https://huggingface.co/datasets/Qiskit/qiskit_humaneval_hard) | 151 | Natural-language instruction: the model writes the whole function | Apache-2.0 |
| [Qiskit HumanEval](https://huggingface.co/datasets/Qiskit/qiskit_humaneval) | 151 | Code stub with a docstring | Apache-2.0 |
| [Qiskit QuantumKatas](https://huggingface.co/datasets/Qiskit/Qiskit-QuantumKatas) | 350 | Task description plus a function stub | **CC BY-NC-SA 4.0 (non-commercial)** |

```bash
mkdir -p data
# Qiskit HumanEval hard (and, the same way, Qiskit/qiskit_humaneval)
uv run --with datasets python -c "from datasets import load_dataset; load_dataset('Qiskit/qiskit_humaneval_hard', split='test').to_json('data/qiskit_humaneval_hard.jsonl', force_ascii=False)"
# Qiskit QuantumKatas
curl -L -o data/qiskit_quantumkatas.jsonl https://huggingface.co/datasets/Qiskit/Qiskit-QuantumKatas/resolve/main/qiskit_quantumkatas.jsonl
```

Check each dataset's license before redistributing it, or any results generated from it: results generated from QuantumKatas fall under CC BY-NC-SA 4.0.

Some Qiskit HumanEval tasks call IBM Quantum services: they can only pass with a saved IBM Quantum account (`QiskitRuntimeService.save_account(...)`); without one, even their canonical solutions fail. The remaining canonical solutions, and all 350 QuantumKatas ones, pass the test-suite validator.

#### LintQ Setup

LintQ is a set of [CodeQL](https://codeql.github.com/) queries, so it needs the CodeQL CLI and a checkout of the LintQ repository:

1. Install the CodeQL CLI and put `codeql` on your `PATH`, e.g. `brew install codeql` on macOS, or download the CodeQL bundle from the [codeql-action releases](https://github.com/github/codeql-action/releases). Check it with `codeql version`.
2. Clone LintQ and install its query pack dependencies:

   ```bash
   git clone --depth 1 https://github.com/sola-st/LintQ.git
   (cd LintQ/qlint/codeql/src && codeql pack install)
   ```

3. Point `LINTQ_DIR` at the checkout:

   ```bash
   export LINTQ_DIR="$(pwd)/LintQ"
   ```


### Benchmark Results

All runs use `MultiTurnStrategy` with a loop budget of 3 (the first attempt plus up to 2 repairs).

#### gpt-oss-120b (test suite + LintQ)

These runs validated the test suite and LintQ.

| Metric | QHE hard: first pass | QHE hard: post repair | QuantumKatas: first pass | QuantumKatas: post repair |
|--------|------|------|------|------|
| Tasks | 151 | 151 | 350 | 350 |
| Passed both validators | 24/151 (16%) | **70/151 (46%)** | 25/350 (7%) | **163/350 (47%)** |
| Failed | 127/151 (84%) | 81/151 (54%) | 325/350 (93%) | 187/350 (53%) |
| Failed the test suite | 113 | 77 | 297 | 148 |
| Flagged by LintQ | 14 | 4 | 28 | 39 |
| LintQ warnings raised | 17 | 4 | 30 | 47 |
| Fixed by the repair loop | — | 46 | — | 138 |

The repair loop roughly triples the Qiskit HumanEval hard pass rate and multiplies the QuantumKatas one by more than six. On QuantumKatas, LintQ flags more tasks after repair than before: a task that fails its tests is never linted, and many programs only start running (and so get linted) after repair.


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

### LintQ Not Configured
```
KeyError: 'LINTQ_DIR'
```
**Solution**: Follow [LintQ Setup](#lintq-setup) and `export LINTQ_DIR=/path/to/LintQ`.

### CodeQL Errors
```
subprocess.CalledProcessError: Command '['codeql', 'database', ...]' returned non-zero exit status
```
**Solution**: Check `codeql version` works, and that `codeql pack install` was run in `$LINTQ_DIR/qlint/codeql/src`.

### Import Error: flake8-qiskit-migration
```
ModuleNotFoundError: No module named 'flake8_qiskit_migration'
```
**Solution**: Use `uv run` which auto-installs dependencies

