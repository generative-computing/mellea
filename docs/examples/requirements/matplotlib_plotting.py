# pytest: unit
"""Examples for matplotlib-specific requirements validation.

This example demonstrates how to use MatplotlibHeadlessBackend, PlotFileSaved,
and PlotDependenciesAvailable requirements to validate code that generates plots.
"""

from mellea.core import ModelOutputThunk
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.plotting import (
    MatplotlibHeadlessBackend,
    PlotDependenciesAvailable,
    PlotFileSaved,
)


def example_1_headless_backend_validation():
    """Validate that matplotlib is configured with a headless backend."""
    print("\n=== Example 1: Headless Backend Validation ===")

    # Code with valid headless backend (Agg)
    valid_code = """```python
import matplotlib
matplotlib.use('Agg')  # Headless backend suitable for servers
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Simple Plot')
```"""

    context = ChatContext().add(ModelOutputThunk(value=valid_code))

    # Validate using the requirement
    req = MatplotlibHeadlessBackend()
    result = req.validation_fn(context)

    print(f"Result: {result.as_bool()}")
    print(f"Reason: {result.reason}")


def example_2_headless_backend_failure():
    """Demonstrate failure when using interactive backend."""
    print("\n=== Example 2: Interactive Backend (Should Fail) ===")

    # Code with interactive backend (TkAgg)
    invalid_code = """```python
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend - requires display
import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
plt.show()
```"""

    context = ChatContext().add(ModelOutputThunk(value=invalid_code))

    req = MatplotlibHeadlessBackend()
    result = req.validation_fn(context)

    print(f"Result: {result.as_bool()}")
    print(f"Reason: {result.reason}")


def example_3_plot_file_saved():
    """Validate that plot is explicitly saved to a file."""
    print("\n=== Example 3: Plot File Saved ===")

    code_with_savefig = """```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 4, 9])
plt.title('My Plot')
plt.savefig('/tmp/output_plot.png')
```"""

    context = ChatContext().add(ModelOutputThunk(value=code_with_savefig))

    # Require plot to be saved to specific path
    req = PlotFileSaved(output_path="/tmp/output_plot.png")
    result = req.validation_fn(context)

    print(f"Result: {result.as_bool()}")
    print(f"Reason: {result.reason}")


def example_4_plot_file_saved_failure():
    """Demonstrate failure when plot is not saved."""
    print("\n=== Example 4: Plot Not Saved (Should Fail) ===")

    code_without_savefig = """```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 4, 9])
plt.title('My Plot')
plt.show()  # Only displays, doesn't save
```"""

    context = ChatContext().add(ModelOutputThunk(value=code_without_savefig))

    req = PlotFileSaved(output_path="/tmp/output_plot.png")
    result = req.validation_fn(context)

    print(f"Result: {result.as_bool()}")
    print(f"Reason: {result.reason}")


def example_5_dependencies_available():
    """Validate that required plotting dependencies are available."""
    print("\n=== Example 5: Dependencies Available ===")

    # Any code containing imports
    code = """```python
import matplotlib
import numpy as np

data = np.array([1, 2, 3, 4, 5])
print(data)
```"""

    context = ChatContext().add(ModelOutputThunk(value=code))

    req = PlotDependenciesAvailable()
    result = req.validation_fn(context)

    print(f"Result: {result.as_bool()}")
    print(f"Reason: {result.reason}")


def example_6_multiple_savefig_calls():
    """Demonstrate validation with multiple savefig calls."""
    print("\n=== Example 6: Multiple Savefig Calls ===")

    code = """```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create first plot
fig1, ax1 = plt.subplots()
ax1.plot([1, 2, 3], [1, 2, 3])
fig1.savefig('/tmp/plot1.png')

# Create second plot
fig2, ax2 = plt.subplots()
ax2.plot([1, 2, 3], [3, 2, 1])
fig2.savefig('/tmp/plot2.png')
```"""

    context = ChatContext().add(ModelOutputThunk(value=code))

    # Check that first plot is saved
    req1 = PlotFileSaved(output_path="/tmp/plot1.png")
    result1 = req1.validation_fn(context)

    # Check that second plot is saved
    req2 = PlotFileSaved(output_path="/tmp/plot2.png")
    result2 = req2.validation_fn(context)

    print(f"Plot 1 saved: {result1.as_bool()} - {result1.reason}")
    print(f"Plot 2 saved: {result2.as_bool()} - {result2.reason}")


def example_7_fig_savefig():
    """Demonstrate validation with fig.savefig() pattern."""
    print("\n=== Example 7: Figure savefig() ===")

    code = """```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter([1, 2, 3], [1, 4, 9])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
fig.savefig('/tmp/scatter.png')
```"""

    context = ChatContext().add(ModelOutputThunk(value=code))

    req = PlotFileSaved(output_path="/tmp/scatter.png")
    result = req.validation_fn(context)

    print(f"Result: {result.as_bool()}")
    print(f"Reason: {result.reason}")


def example_8_keyword_arguments():
    """Demonstrate validation with keyword arguments in savefig."""
    print("\n=== Example 8: Keyword Arguments ===")

    code = """```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3])
fig.savefig(fname='/tmp/my_plot.png', dpi=300, bbox_inches='tight')
```"""

    context = ChatContext().add(ModelOutputThunk(value=code))

    req = PlotFileSaved(output_path="/tmp/my_plot.png")
    result = req.validation_fn(context)

    print(f"Result: {result.as_bool()}")
    print(f"Reason: {result.reason}")


if __name__ == "__main__":
    example_1_headless_backend_validation()
    example_2_headless_backend_failure()
    example_3_plot_file_saved()
    example_4_plot_file_saved_failure()
    example_5_dependencies_available()
    example_6_multiple_savefig_calls()
    example_7_fig_savefig()
    example_8_keyword_arguments()
