# pytest: ollama, e2e, qualitative
"""Repair plotting code with Python-tool and plotting-specific requirements."""

import asyncio
import tempfile
import traceback
from pathlib import Path

import mellea
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import (
    python_plotting_requirements,
    python_tool_requirements,
)
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
    """Run the plotting repair example."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "plot.png")

        m = mellea.start_session()

        requirements = [
            *python_tool_requirements(allowed_imports=["numpy", "matplotlib", "math"]),
            *python_plotting_requirements(output_path=output_path),
        ]

        sampling_strategy = SOFAISamplingStrategy(
            s1_solver_backend=m.backend,
            s2_solver_backend=m.backend,
            s2_solver_mode="fresh_start",
            loop_budget=3,
            feedback_strategy="first_error",
        )

        task_summary = (
            f"Create a plot of sin(x) for x in 0..2π and save it to {output_path}"
        )
        description = f"""{task_summary}

Requirements:
- Use the python tool to execute your code
- Import numpy and matplotlib
- Generate x values from 0 to 2π
- Plot sin(x) against x
- Save the plot to the specified file path

Use the python tool with your complete code."""
        instruction = Instruction(description=description)

        ctx = ChatContext()

        print("=" * 70)
        print("Testing plotting-code repair with Python tool requirements")
        print("=" * 70)
        print(f"Task: {task_summary}\n")

        try:
            result = await sampling_strategy.sample(
                action=instruction,
                context=ctx,
                backend=m.backend,
                requirements=requirements,
                tool_calls=True,
                model_options={ModelOption.TOOLS: [MelleaTool.from_callable(python)]},
            )

            print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}\n")

            if result.success:
                print("✓ Model successfully generated and executed plotting code")
                print("\nFinal generated code:")
                print("-" * 70)
                print(result.result.value)
                print("-" * 70)

                if Path(output_path).exists():  # noqa: ASYNC240
                    file_size = Path(output_path).stat().st_size  # noqa: ASYNC240
                    print(f"\n✓ Output file created: {output_path}")
                    print(f"  File size: {file_size} bytes")
                else:
                    print(f"\n✗ Output file not found: {output_path}")

                print(f"\nRepair iterations: {len(result.sample_validations)}")
                for attempt_idx, validations in enumerate(result.sample_validations, 1):
                    passed = sum(1 for _, val in validations if val.as_bool())
                    total = len(validations)
                    status = "✓" if passed == total else "✗"
                    print(
                        f"  {status} Attempt {attempt_idx}: {passed}/{total} "
                        f"requirements passed"
                    )

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
            traceback.print_exc()

        print("\n" + "=" * 70)
        print("Test completed")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

# Made with Bob
