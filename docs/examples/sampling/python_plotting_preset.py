# pytest: e2e, qualitative, ollama
"""Example demonstrating python_plotting_sampling() preset.

Shows how the plotting preset bundles Python code generation requirements with
matplotlib-specific constraints for headless rendering and file output validation.

The preset validates:
1. Python code extraction and syntax
2. Matplotlib headless backend configuration (matplotlib.use('Agg'))
3. Plot file output (savefig() call with expected path)
4. Code execution with sandbox isolation (enabled by default)
5. Import restrictions and output size limits

Plotting typically requires more repair iterations and stricter sandboxing than
general code generation, which this preset reflects in its defaults:
- loop_budget=3 (vs. 2 for code generation)
- use_sandbox=True (vs. False for code generation)
- timeout_seconds=10 (vs. 5 for code generation)
"""

import tempfile

import mellea
from mellea.stdlib.sampling import python_plotting_sampling


def example_scatter_plot():
    """Generate matplotlib code to create a scatter plot.

    Demonstrates the basic usage of python_plotting_sampling() with an expected
    output file. The model generates code to create a scatter plot of random points.

    The preset ensures:
    1. matplotlib.use('Agg') is called for headless rendering
    2. plt.savefig(output_path) is called with the specified path
    3. Code executes without errors in isolated sandbox
    """
    session = mellea.start_session()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/scatter.png"

        preset = python_plotting_sampling(output_path=output_path, use_sandbox=False)

        prompt = (
            f"Write Python code to create a scatter plot of 50 random points. "
            f"Use matplotlib. Save the plot to '{output_path}'. "
            f"Set up the headless backend properly."
        )

        result = session.instruct(
            prompt, requirements=preset.requirements, strategy=preset.strategy
        )

        code = str(result)
        assert "matplotlib" in code.lower(), "Code should import matplotlib"
        assert "scatter" in code.lower(), "Code should use scatter plot"


def example_line_plot_with_labels():
    """Generate matplotlib code with title and axis labels.

    Demonstrates generating more complex plotting code with labels and formatting.
    The preset handles headless backend setup and file output validation automatically.
    """
    session = mellea.start_session()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/line_plot.png"

        preset = python_plotting_sampling(output_path=output_path, use_sandbox=False)

        prompt = (
            f"Write Python code to create a line plot showing y = x^2 for x in range(0, 10). "
            f"Add title 'Quadratic Function', xlabel 'x', ylabel 'y'. "
            f"Save to '{output_path}'. Use proper headless matplotlib setup."
        )

        result = session.instruct(
            prompt, requirements=preset.requirements, strategy=preset.strategy
        )

        code = str(result)
        assert "plot" in code.lower(), "Code should create a plot"
        assert "title" in code.lower(), "Code should set title"


def example_without_output_path():
    """Generate plotting code without specifying output path.

    Demonstrates using the preset when you don't care about a specific output path.
    The preset validates that matplotlib dependencies are available but doesn't
    enforce a specific savefig() call.
    """
    session = mellea.start_session()

    # No output_path specified — validates dependencies only
    preset = python_plotting_sampling()

    prompt = (
        "Write Python code to create a histogram of 1000 random normal values. "
        "Use matplotlib with headless backend. Set up the plot properly."
    )

    result = session.instruct(
        prompt, requirements=preset.requirements, strategy=preset.strategy
    )

    code = str(result)
    assert "hist" in code.lower(), "Code should use histogram"
    assert "matplotlib" in code.lower(), "Code should use matplotlib"


def example_with_import_restrictions():
    """Generate plotting code with restricted imports.

    Demonstrates combining import restrictions with plotting requirements.
    Only matplotlib and numpy are allowed (plus standard library).
    """
    session = mellea.start_session()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/restricted.png"

        preset = python_plotting_sampling(
            output_path=output_path,
            allowed_imports=["matplotlib", "numpy"],
            use_sandbox=False,
            loop_budget=7,
        )

        prompt = (
            f"Create a bar plot using matplotlib and numpy. "
            f"Plot bars for [1, 2, 3, 4, 5] with heights [10, 24, 36, 18, 7]. "
            f"Save to '{output_path}'. "
            f"Only use matplotlib and numpy (no other external modules)."
        )

        result = session.instruct(
            prompt, requirements=preset.requirements, strategy=preset.strategy
        )

        code = str(result)
        assert "bar" in code.lower(), "Code should use bar plot"
        # Should not have forbidden imports
        assert "subprocess" not in code
        assert "requests" not in code


def example_with_custom_parameters():
    """Generate plotting code with custom preset parameters.

    Demonstrates fine-tuning the preset's parameters:
    - loop_budget: More repair attempts for complex plots
    - timeout_seconds: Longer timeout for rendering-heavy plots
    - use_sandbox: Keep sandbox enabled for security
    """
    session = mellea.start_session()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/complex.png"

        preset = python_plotting_sampling(
            output_path=output_path,
            loop_budget=7,  # More repair attempts for complex plots
            timeout_seconds=15,  # Longer timeout
            use_sandbox=False,
        )

        prompt = (
            f"Create a complex matplotlib visualization with multiple subplots: "
            f"(1) histogram, (2) scatter plot, (3) line plot. "
            f"Save to '{output_path}'. Use headless backend."
        )

        result = session.instruct(
            prompt, requirements=preset.requirements, strategy=preset.strategy
        )

        code = str(result)
        assert "subplot" in code.lower() or "figure" in code.lower()


def main():
    """Run all examples."""
    print("Running example_scatter_plot...")
    example_scatter_plot()
    print("✓ Scatter plot example passed\n")

    print("Running example_line_plot_with_labels...")
    example_line_plot_with_labels()
    print("✓ Line plot with labels example completed\n")

    print("Running example_without_output_path...")
    example_without_output_path()
    print("✓ Plot without output path example completed\n")

    print("Running example_with_import_restrictions...")
    example_with_import_restrictions()
    print("✓ Import restrictions example completed\n")

    print("Running example_with_custom_parameters...")
    example_with_custom_parameters()
    print("✓ Custom parameters example completed\n")

    print("All plotting examples completed!")


if __name__ == "__main__":
    main()
