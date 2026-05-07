# pytest: ollama, e2e, qualitative
"""Granite 4.1 repairs the three canonical plotting failures with Python tool.

This example demonstrates:
1. Creating a PythonToolRequirements bundle for plotting validation
2. Using SOFAI sampling strategy with repair feedback loop
3. Granite 4.1 repairing through: syntax → imports → headless backend → savefig

Canonical task: "Create a plot of sin(x) for x in 0..2π and save to /tmp/plot.png"

The model will encounter and repair:
- Attempt 1: Missing matplotlib.use('Agg') (non-headless backend)
- Attempt 2: Missing plt.savefig() call
- Attempt 3: Success with both fixes applied

The requirements bundle provides actionable failure messages that guide the model
through each repair iteration without explicit instruction.
"""

import mellea
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import PythonToolRequirements
from mellea.stdlib.sampling import SOFAISamplingStrategy
from mellea.stdlib.tools import local_code_interpreter
from mellea.stdlib.tools.interpreter import ExecutionResult


def python(code: str) -> ExecutionResult:
    """Execute Python code.

    Args:
        code: Python code to execute

    Returns:
        Execution result containing stdout, stderr, and success status
    """
    return local_code_interpreter(code)


async def main():
    """Run the canonical plotting repair example."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "plot.png")

        # Initialize session with local backend
        m = mellea.start_session()

        # Create requirements bundle for plotting validation
        # Allows matplotlib import (no output_path = skip file creation check)
        bundle = PythonToolRequirements(allowed_imports=["numpy", "matplotlib", "math"])

        # Define SOFAI strategy for repair: S1 (fast) up to 3 times, then S2 (slow)
        sampling_strategy = SOFAISamplingStrategy(
            s1_solver_backend=m.backend,
            s2_solver_backend=m.backend,
            s2_solver_mode="fresh_start",
            loop_budget=3,
            feedback_strategy="first_error",
        )

        # Create the plotting task instruction
        description = f"""Create a plot of sin(x) for x in 0..2π and save it to {output_path}.

Requirements:
- Use the python tool to execute your code
- Import numpy and matplotlib
- Generate x values from 0 to 2π
- Plot sin(x) against x
- Save the plot to the specified file path

Use the python tool with your complete code."""
        instruction = Instruction(description=description)

        # Create a chat context for multi-turn repair
        ctx = ChatContext()

        print("=" * 70)
        print("Testing Granite 4.1's ability to repair plotting failures")
        print("=" * 70)
        print(f"Task: Create a plot of sin(x) and save to {output_path}\n")

        try:
            # Run the sampling strategy with requirements
            result = await sampling_strategy.sample(
                action=instruction,
                context=ctx,
                backend=m.backend,
                requirements=bundle.requirements,
                tool_calls=True,
                model_options={ModelOption.TOOLS: [MelleaTool.from_callable(python)]},
            )

            print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}\n")

            if result.success:
                print("✓ Granite 4.1 successfully generated and executed plotting code")
                print("\nFinal generated code:")
                print("-" * 70)
                print(result.result.value)
                print("-" * 70)

                # Verify output file exists
                from pathlib import Path

                if Path(output_path).exists():  # noqa: ASYNC240
                    file_size = Path(output_path).stat().st_size  # noqa: ASYNC240
                    print(f"\n✓ Output file created: {output_path}")
                    print(f"  File size: {file_size} bytes")
                else:
                    print(f"\n✗ Output file not found: {output_path}")

                # Print repair history
                print(f"\nRepair iterations: {len(result.sample_validations)}")
                for attempt_idx, validations in enumerate(result.sample_validations, 1):
                    passed = sum(1 for _, val in validations if val.as_bool())
                    total = len(validations)
                    status = "✓" if passed == total else "✗"
                    print(
                        f"  {status} Attempt {attempt_idx}: {passed}/{total} "
                        f"requirements passed"
                    )

                    # Show which requirements failed
                    for req, val in validations:
                        if not val.as_bool():
                            print(f"     - {req.description}")
                            if val.reason:
                                reason_preview = val.reason[:100].replace("\n", " ")
                                print(f"       Error: {reason_preview}...")

            else:
                print("✗ Failed to generate working plotting code after all attempts\n")
                print("Last attempt output:")
                print("-" * 70)
                print(result.result.value)
                print("-" * 70)

                # Print failure history
                print(f"\nFailure history ({len(result.sample_validations)} attempts):")
                for attempt_idx, validations in enumerate(result.sample_validations, 1):
                    failed_count = sum(1 for _, val in validations if not val.as_bool())
                    if failed_count > 0:
                        print(f"\n  Attempt {attempt_idx}:")
                        for req, val in validations:
                            if not val.as_bool():
                                print(f"    - {req.description}")
                                if val.reason:
                                    reason_lines = val.reason.split("\n")[:2]
                                    for line in reason_lines:
                                        print(f"      {line}")

        except Exception as e:
            print(f"✗ Exception during sampling: {e}")
            import traceback

            traceback.print_exc()

        print("\n" + "=" * 70)
        print("Test completed")
        print("=" * 70)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
