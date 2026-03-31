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
4. **Repair loop**: Automatically repair code that fails validation (up to 10 attempts)

Requirements:
    - flake8-qiskit-migration: Installed automatically when run via `uv run`
    - Ollama backend running with a compatible model (e.g., mistral-small-3.2-24b-qiskit-GGUF)

Example:
    Run as a standalone script (dependencies installed automatically):
        $ uv run docs/examples/instruct_validate_repair/qiskit_code_validation/qiskit_code_validation.py
"""

import time

from validation_helpers import validate_input_code, validate_qiskit_migration

from mellea import MelleaSession, start_session
from mellea.backends import ModelOption
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import MultiTurnStrategy, RepairTemplateStrategy

# Optional system prompt for models not specialized for Qiskit.
# Set system_prompt = QISKIT_SYSTEM_PROMPT in test_qiskit_code_validation() to enable.
QISKIT_SYSTEM_PROMPT = """\
You are the Qiskit code assistant, a Qiskit coding expert developed by IBM Quantum. \
Your mission is to help users write good Qiskit code and advise them on best practices \
for quantum computing using Qiskit and IBM Quantum and its hardware and services. \
You stick to the user request, without adding non-requested information or yapping.

When doing code generation, you always generate Python and Qiskit code. If the input \
you received only contains code, your task is to complete the code without adding extra \
explanations or text.

The current version of `qiskit` is `2.1`. Ensure your code is valid Python and Qiskit. \
The official documentation is available at https://quantum.cloud.ibm.com/docs/en. \
Avoid `https://qiskit.org` links as they are not active.

Code standards — never use deprecated methods:
- Transpilation: use `generate_preset_pass_manager()` instead of `transpile()`
- Execution: use `SamplerV2` or `EstimatorV2` primitives instead of `execute()`
- Provider: `qiskit-ibmq-provider` / `IBMQ` was deprecated in 2023; use `qiskit-ibm-runtime` instead
- Simulator: import as `from qiskit_aer import AerSimulator`, not `from qiskit.providers.aer import AerSimulator`
- Random circuits: import as `from qiskit.circuit.random import random_circuit`

When no backend is specified, default to `ibm_fez`, `ibm_marrakesh`, `ibm_pittsburg`, or `ibm_kingston`. \
Avoid simulators unless explicitly requested.

The four steps of a Qiskit pattern: (1) Map problem to quantum circuits and operators. \
(2) Optimize for target hardware. (3) Execute on target hardware. (4) Post-process results.
"""


def generate_validated_qiskit_code(
    m: MelleaSession,
    prompt: str,
    strategy: MultiTurnStrategy | RepairTemplateStrategy,
    *,
    system_prompt: str | None = None,
    grounding_context: dict[str, str] | None = None,
) -> tuple[str, bool, int]:
    """Generate Qiskit code that passes Qiskit migration validation.

    This function implements the Instruct-Validate-Repair pattern:
    1. Pre-validates input code
    2. Instructs the LLM with structured requirements
    3. Validates output against QKT rules
    4. Repairs code if validation fails (up to the strategy's loop_budget times)

    Args:
        m: Mellea session
        prompt: User prompt for code generation
        strategy: Sampling strategy for handling validation failures
        system_prompt: Optional system prompt passed via ModelOption.SYSTEM_PROMPT
        grounding_context: Optional grounding context dict passed to m.instruct()

    Returns:
        Tuple of (generated_code, success, attempts_used)
    """
    # Only pass optional kwargs if they have values — avoids passing None to m.instruct()
    extra: dict = {}
    if grounding_context:
        extra["grounding_context"] = grounding_context
    if system_prompt:
        extra["model_options"] = {ModelOption.SYSTEM_PROMPT: system_prompt}

    # Generate code with output validation only
    code_candidate = m.instruct(
        prompt,
        requirements=[
            req(
                "Code must pass Qiskit migration validation (QKT rules)",
                validation_fn=simple_validate(validate_qiskit_migration),
            )
        ],
        strategy=strategy,
        return_sampling_results=True,
        **extra,
    )

    attempts = (
        len(code_candidate.sample_generations)
        if code_candidate.sample_generations
        else 1
    )

    if code_candidate.success:
        return str(code_candidate.result), True, attempts
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
            return (
                str(code_candidate.sample_generations[-1].value or ""),
                False,
                attempts,
            )
        print("No code generations available")
        return "", False, attempts


def test_qiskit_code_validation() -> None:
    """Test Qiskit code validation with deprecated code that needs fixing.

    This test demonstrates the IVR pattern by providing deprecated Qiskit code
    that uses old APIs (BasicAer, execute) and having the LLM fix it to use
    modern Qiskit APIs that pass QKT validation rules.
    """
    # Model — requires Ollama with the model pulled locally
    # See README.md for model options and tradeoffs
    model_id = "hf.co/Qiskit/mistral-small-3.2-24b-qiskit-GGUF:latest"

    # System prompt — None uses the model's built-in Qiskit knowledge (default)
    # Set to QISKIT_SYSTEM_PROMPT when using a model not specialized for Qiskit
    system_prompt = None

    # Prompt - replace with your own or see README.md for examples
    prompt = """from qiskit import BasicAer, QuantumCircuit, execute

backend = BasicAer.get_backend('qasm_simulator')

qc = QuantumCircuit(5, 5)
qc.h(0)
qc.cnot(0, range(1, 5))
qc.measure_all()

# run circuit on the simulator
"""

    print("\n====== Prompt ======")
    print(prompt)
    print("======================\n")

    # Strategy selection - True for MultiTurnStrategy, False for RepairTemplateStrategy
    # MultiTurnStrategy: Adds validation failure reasons as a new user message in the conversation
    # RepairTemplateStrategy: Adds validation failure reasons to the instruction and retries
    use_multiturn_strategy = False

    # Initialize the required context
    ctx = ChatContext() if use_multiturn_strategy else SimpleContext()
    if use_multiturn_strategy:
        strategy: MultiTurnStrategy | RepairTemplateStrategy = MultiTurnStrategy(
            loop_budget=10
        )
    else:
        strategy = RepairTemplateStrategy(loop_budget=10)

    with start_session(
        model_id=model_id,
        backend_name="ollama",
        ctx=ctx,
        model_options={ModelOption.TEMPERATURE: 0.8, ModelOption.MAX_NEW_TOKENS: 2048},
    ) as m:
        start_time = time.time()

        code, success, attempts = generate_validated_qiskit_code(
            m, prompt, strategy, system_prompt=system_prompt
        )
        elapsed = time.time() - start_time

    print(f"\n====== Result ({elapsed:.1f}s, {attempts} attempt(s)) ======")
    print(code)
    print("======================\n")

    if success:
        print("✓ Code passes Qiskit migration validation")
    else:
        _, error_msg = validate_qiskit_migration(code)
        print("✗ Validation errors:")
        print(error_msg)


if __name__ == "__main__":
    # Run the example when executed as a script
    test_qiskit_code_validation()
