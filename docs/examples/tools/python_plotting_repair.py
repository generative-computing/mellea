# pytest: ollama, e2e, qualitative
"""Repair plotting code with Python-tool and plotting-specific requirements.

This example demonstrates the full tool lifecycle:
1. Model generates code and creates tool calls
2. Sampling validation checks code quality without execution
3. Tool is explicitly invoked after sampling succeeds (via _call_tools)
4. Results are returned to caller for inspection/handling

Key insight: Tool execution is explicit and controlled by the caller,
not automatic within the sampling pipeline. This allows fine-grained control
over when/if tools are invoked, and enables safety checks (see tool_hooks.py).

Prerequisites:
    matplotlib must be installed for code execution to succeed:
    $ uv pip install matplotlib
"""

import tempfile
import traceback
from pathlib import Path

import mellea
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool
from mellea.stdlib.functional import _call_tools
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


def main():
    """Run the plotting repair example."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "plot.png")

        m = mellea.start_session(context_type="chat")

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

        print("=" * 70)
        print("Testing plotting-code repair with Python tool requirements")
        print("=" * 70)
        print(f"Task: {task_summary}\n")

        try:
            result = m.instruct(
                task_summary,
                requirements=requirements,
                strategy=sampling_strategy,
                return_sampling_results=True,
                tool_calls=True,
                model_options={ModelOption.TOOLS: [MelleaTool.from_callable(python)]},
            )

            print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}\n")

            if result.success:
                print("✓ Model successfully generated plotting code")

                # Invoke the generated tools from the final result
                if (
                    result.result
                    and hasattr(result.result, "tool_calls")
                    and result.result.tool_calls
                ):
                    # Print the generated code
                    for tool_name, tool_call in result.result.tool_calls.items():
                        if tool_call.args and "code" in tool_call.args:
                            code = tool_call.args["code"]
                            print(f"\n{'=' * 70}")
                            print(f"Generated Python code for tool '{tool_name}':")
                            print(f"{'=' * 70}")
                            print(code)
                            print(f"{'=' * 70}\n")

                    tool_outputs = _call_tools(result.result, m.backend)

                    if tool_outputs:
                        print("✓ Tool executed successfully")
                        for i, output in enumerate(tool_outputs, 1):
                            print(f"  Output {i}: {output.content}")
                else:
                    print("ℹ No tool calls in final result")

                print(f"\nCode saved to: {output_path}")

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
    main()

# Made with Bob
