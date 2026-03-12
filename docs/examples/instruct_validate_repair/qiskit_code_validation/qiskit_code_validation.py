# pytest: ollama, llm, qualitative, skip
# /// script
# dependencies = [
#   "mellea",
#   "flake8-qiskit-migration",
# ]
# ///
"""Qiskit Code Validation with Instruct-Validate-Repair Pattern.

This example demonstrates using Mellea's Instruct-Validate-Repair (IVR) pattern
to generate Qiskit quantum computing code that automatically passes
flake8-qiskit-migration validation rules (QKT rules).

The pipeline follows these steps:
1. **Pre-condition validation**: Validate prompt content and any input code
2. **Instruction**: LLM generates code following structured requirements
3. **Post-condition validation**: Validate generated code against QKT rules
4. **Repair loop**: Automatically repair code that fails validation (up to 5 attempts)

Requirements:
    - flake8-qiskit-migration: Installed automatically when run via `uv run`
    - Ollama backend running with a compatible model (e.g., mistral-small-3.2-24b-qiskit-GGUF)

Example:
    Run as a standalone script (dependencies installed automatically):
        $ uv run docs/examples/instruct_validate_repair/qiskit_code_validation/qiskit_code_validation.py
"""

import time
from typing import Literal

from validation_helpers import validate_input_code, validate_qiskit_migration

import mellea
from mellea.backends import ModelOption
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RepairTemplateStrategy


def generate_validated_qiskit_code(
    m: mellea.MelleaSession, prompt: str, max_repair_attempts: int = 5
) -> str:
    """Generate Qiskit code that passes Qiskit migration validation.

    This function implements the Instruct-Validate-Repair pattern:
    1. Pre-validates input code
    2. Instructs the LLM with structured requirements
    3. Validates output against QKT rules
    4. Repairs code if validation fails (up to max_repair_attempts times)

    Args:
        m: Mellea session
        prompt: User prompt for code generation
        max_repair_attempts: Maximum number of repair attempts for validation failures

    Returns:
        Generated code that passes validation

    Raises:
        ValueError: If prompt validation fails
    """
    # Pre-validate input code if present — include violations as context rather than failing
    is_valid, error_msg = validate_input_code(prompt)
    input_code_errors = None
    if not is_valid:
        print(
            f"Input code has QKT violations, including as context for LLM: {error_msg}"
        )
        input_code_errors = error_msg

    # Build the instruction prompt, optionally augmented with input code violations
    instruct_prompt = prompt
    if input_code_errors is not None:
        instruct_prompt = (
            f"{prompt}\n\n"
            f"Note: the code above has the following Qiskit migration issues that must be fixed:\n"
            f"{input_code_errors}"
        )

    # Generate code with output validation only
    code_candidate = m.instruct(
        instruct_prompt,
        requirements=[
            req(
                "Code must pass Qiskit migration validation (QKT rules)",
                validation_fn=simple_validate(validate_qiskit_migration),
            )
        ],
        strategy=RepairTemplateStrategy(loop_budget=max_repair_attempts),
        return_sampling_results=True,
    )

    if code_candidate.success:
        return str(code_candidate.result)
    else:
        print("Code generation did not fully succeed, returning best attempt")
        # Log detailed validation failure reasons
        if code_candidate.result_validations:
            for requirement, validation_result in code_candidate.result_validations:
                if not validation_result:
                    print(
                        f"  Failed requirement: {requirement.description} — {validation_result.reason}"
                    )
        # Return best attempt even if validation failed
        if code_candidate.sample_generations:
            return str(code_candidate.sample_generations[0].value or "")
        print("No code generations available")
        return ""


# Mellea IVR loop budget
MAX_REPAIR_ATTEMPTS = 5

# Mellea accessible backend
DEFAULT_BACKEND: Literal["ollama", "hf", "openai", "watsonx", "litellm"] = "ollama"

# DEFAULT_MODEL_ID = "granite4:micro-h"
# DEFAULT_MODEL_ID = "granite4:small-h"
DEFAULT_MODEL_ID = "hf.co/Qiskit/mistral-small-3.2-24b-qiskit-GGUF:latest"

# Model options to play around with
MODEL_OPTIONS = {ModelOption.TEMPERATURE: 0.8, ModelOption.MAX_NEW_TOKENS: 2048}


# Set a prompt or uncomment one of the sample prompts

# PROMPT = "create a bell state circuit"
# PROMPT = "use qiskit to list fake backends"
# PROMPT = "give me a random qiskit circuit"

###############################################################################

# PROMPT = """Complete this code:
# ```python
# from qiskit import QuantumCircuit

# qc = QuantumCircuit(3)
# qc.toffoli(0, 1, 2)

# # draw the circuit
# ```
# """

###############################################################################

# PROMPT = """from qiskit import QuantumCircuit
# from qiskit_ibm_runtime import QiskitRuntimeService

# # define a Bell circuit and run it in ibm_salamanca using QiskitRuntimeService"""

###############################################################################

# PROMPT = """from qiskit.circuit.random import random_circuit
# from qiskit.quantum_info import SparsePauliOp
# from qiskit_ibm_runtime import Estimator, Options, QiskitRuntimeService, Session

# # create a Qiskit random circuit named "circuit" with 2 qubits, depth 2, seed 1.
# # After that, generate an observable type SparsePauliOp("IY"). Run it in the backend "ibm_sherbrooke" using QiskitRuntimeService inside a session
# # Instantiate the runtime Estimator primitive using the session and the options optimization level 3 and resilience level 2. Run the estimator
# # Conclude the code printing the observable, expectation value and the metadata of the job."""

###############################################################################

# PROMPT = """from qiskit import QuantumCircuit

# # create an entanglement state circuit
# """

###############################################################################

PROMPT = """from qiskit import BasicAer, QuantumCircuit, execute

backend = BasicAer.get_backend('qasm_simulator')

qc = QuantumCircuit(5, 5)
qc.h(0)
qc.cnot(0, range(1, 5))
qc.measure_all()

# run circuit on the simulator
"""


def test_qiskit_code_validation() -> None:
    """Test Qiskit code validation with deprecated code that needs fixing.

    This test demonstrates the IVR pattern by providing deprecated Qiskit code
    that uses old APIs (BasicAer, execute) and having the LLM fix it to use
    modern Qiskit APIs that pass QKT validation rules.
    """
    print("\n====== Prompt ======")
    print(PROMPT)
    print("======================\n")

    with mellea.start_session(
        model_id=DEFAULT_MODEL_ID,
        backend_name=DEFAULT_BACKEND,
        model_options=MODEL_OPTIONS,
    ) as m:
        start_time = time.time()
        code = generate_validated_qiskit_code(m, PROMPT, MAX_REPAIR_ATTEMPTS)
        elapsed = time.time() - start_time

    print(f"\n====== Result ({elapsed:.1f}s) ======")
    print(code)
    print("======================\n")

    # Validate the generated code
    is_valid, error_msg = validate_qiskit_migration(code)

    if is_valid:
        print("✓ Code passes Qiskit migration validation")
    else:
        print("✗ Validation errors:")
        print(error_msg)


if __name__ == "__main__":
    # Run the example when executed as a script
    test_qiskit_code_validation()
