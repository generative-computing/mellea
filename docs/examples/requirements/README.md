# Requirements Examples

This directory contains examples for using Mellea's requirement validation system, including specialized requirements for RAG (Retrieval-Augmented Generation) workflows and code generation tasks like plotting.

## Files

### matplotlib_plotting.py
Demonstrates how to use matplotlib-specific requirements to validate code that generates plots.

**Key Features:**
- Validating headless backend configuration (Agg, Cairo, pdf, etc.)
- Ensuring plots are explicitly saved to files
- Checking that required dependencies (matplotlib, numpy) are available
- Multiple code patterns: `plt.savefig()`, `fig.savefig()`, `ax.savefig()`
- Supporting both positional and keyword arguments

**Examples Included:**
1. Headless backend validation (valid Agg backend)
2. Headless backend failure (interactive TkAgg backend)
3. Plot file saved validation
4. Plot file saved failure (no savefig)
5. Dependencies available check
6. Multiple savefig calls
7. Figure savefig() pattern
8. Keyword arguments in savefig

## MatplotlibHeadlessBackend

The `MatplotlibHeadlessBackend` requirement validates that matplotlib code uses a headless backend suitable for server environments.

### Basic Usage

```python
from mellea.stdlib.requirements.plotting import MatplotlibHeadlessBackend
from mellea.stdlib.context import ChatContext
from mellea.core import ModelOutputThunk

code = """```python
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
```"""

context = ChatContext().add(ModelOutputThunk(value=code))
req = MatplotlibHeadlessBackend()
result = req.validation_fn(context)
print(result.as_bool())  # True
```

### Supported Backends
- `Agg` — Raster output (most common)
- `Cairo` — Vector output with Cairo
- `pdf`, `svg`, `pgf` — File formats
- `nbAgg` — Jupyter notebooks
- `module://gr.matplotlib.backend_gr` — GR graphics library

## PlotFileSaved

The `PlotFileSaved` requirement validates that plots are explicitly saved to a specific file path.

### Basic Usage

```python
from mellea.stdlib.requirements.plotting import PlotFileSaved
from mellea.stdlib.context import ChatContext
from mellea.core import ModelOutputThunk

code = """```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.savefig('/tmp/plot.png')
```"""

context = ChatContext().add(ModelOutputThunk(value=code))
req = PlotFileSaved(output_path="/tmp/plot.png")
result = req.validation_fn(context)
print(result.as_bool())  # True
```

### Supported Patterns
- `plt.savefig('/tmp/plot.png')`
- `fig.savefig('/tmp/plot.png')`
- `ax.savefig('/tmp/plot.png')`
- Keyword arguments: `fig.savefig(fname='/tmp/plot.png', dpi=300)`

## PlotDependenciesAvailable

The `PlotDependenciesAvailable` requirement validates that matplotlib and numpy are importable.

### Basic Usage

```python
from mellea.stdlib.requirements.plotting import PlotDependenciesAvailable

req = PlotDependenciesAvailable()
result = req.validation_fn(context)
print(result.as_bool())  # True if matplotlib and numpy are available
```

## Use Cases

**When to Use:**
- ✅ Generating plotting code that must run on servers (headless)
- ✅ Validating plots are saved to files (not displayed interactively)
- ✅ Ensuring code can run in CI/CD environments
- ✅ Verifying required data science libraries are available
- ✅ Code quality gates for machine learning notebooks

**When NOT to Use:**
- ❌ Interactive plotting applications
- ❌ Jupyter notebooks meant for display (use `nbAgg` backend instead)
- ❌ Desktop applications with display servers

## Related Documentation

- See `mellea/stdlib/requirements/plotting/matplotlib.py` for implementation
- See `test/stdlib/requirements/plotting/test_matplotlib.py` for tests

## Requirements

- matplotlib and numpy installed (auto-checked by requirements)
- Any code analysis backend (requirements use AST analysis)