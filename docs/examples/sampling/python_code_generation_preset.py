# pytest: e2e, qualitative, ollama
"""Example demonstrating python_code_generation_sampling() preset.

Shows how the preset bundles requirements and feedback into a reusable
configuration for Python code generation tasks, with automatic repair on failure.

The preset handles:
1. Code extraction validation
2. Syntax validation
3. Execution validation with optional sandbox isolation
4. Import restrictions (optional allowlist)
5. Output size limits
6. Model-friendly repair feedback for each failure type
"""

import mellea
from mellea.stdlib.sampling import python_code_generation_sampling


def example_basic_code_generation():
    """Generate Python code with automatic repair feedback.

    Demonstrates the basic usage of python_code_generation_sampling() with
    default parameters. The model generates code to compute the sum of 1 to 100.
    """
    session = mellea.start_session()

    preset = python_code_generation_sampling()

    prompt = (
        "Write Python code to compute the sum of integers from 1 to 100. "
        "Return the result using a variable named 'total'."
    )

    result = session.instruct(
        prompt, requirements=preset.requirements, strategy=preset.strategy
    )

    code = str(result)
    print("Generated code:")
    print(code)
    print()
    assert "100" in code or "5050" in code, "Result should reference the computation"


def example_with_import_restrictions():
    """Generate code with strict import allowlist.

    Demonstrates restricting imports to safe modules (numpy, math).
    If the model tries to import subprocess or other restricted modules,
    the preset provides model-friendly feedback about forbidden imports.
    """
    session = mellea.start_session()

    preset = python_code_generation_sampling(
        allowed_imports=["numpy", "math"],
        loop_budget=3,  # Allow more repair attempts for restricted imports
    )

    prompt = (
        "Write Python code using numpy to create an array [1, 2, 3, 4, 5] "
        "and compute its mean. Only use numpy and math modules."
    )

    result = session.instruct(
        prompt, requirements=preset.requirements, strategy=preset.strategy
    )

    # If code generation fails due to import violations, the feedback will guide
    # the model: "Your code imports forbidden modules: [X]. These are not available.
    # Try: Remove these imports and use only standard library or whitelisted modules."

    code = str(result)
    print("Generated code (with import restrictions):")
    print(code)
    print()
    assert "numpy" in code, "Code should use numpy"
    # Verify no forbidden imports are present
    assert "subprocess" not in code
    assert "os" not in code


def example_with_tighter_constraints():
    """Generate code with stricter resource constraints.

    Demonstrates configuring the preset with:
    - Smaller output limit (5KB instead of 10KB)
    - Shorter timeout (3 seconds instead of 5)
    - Only 2 repair attempts instead of the default

    Useful for resource-constrained environments or performance-critical tasks.
    """
    session = mellea.start_session()

    preset = python_code_generation_sampling(
        output_limit_chars=5000, timeout_seconds=3, loop_budget=2
    )

    prompt = (
        "Write a short Python function that returns the first 10 Fibonacci numbers "
        "as a list. Keep it concise."
    )

    result = session.instruct(
        prompt, requirements=preset.requirements, strategy=preset.strategy
    )

    # Verify output stays within the configured limit
    code = str(result)
    print("Generated code (with tight constraints):")
    print(code)
    print()
    assert len(code) < 5000, "Output should be within limit"


def main():
    """Run all examples."""
    print("Running example_basic_code_generation...")
    example_basic_code_generation()
    print("✓ Basic code generation example passed\n")

    print("Running example_with_import_restrictions...")
    example_with_import_restrictions()
    print("✓ Import restrictions example completed\n")

    print("Running example_with_tighter_constraints...")
    example_with_tighter_constraints()
    print("✓ Tighter constraints example completed\n")

    print("All examples completed!")


if __name__ == "__main__":
    main()
