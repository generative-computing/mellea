# Requirements Examples

This directory contains examples for using Mellea's requirement validation system,
including specialized requirements for RAG (Retrieval-Augmented Generation) workflows
and code generation tasks like plotting.

## Files

### code_generation_and_execution.py

Demonstrates the complete pipeline of code generation, data extraction, and graph
visualization. **NOTE: The default tier is `"static"` (no execution). This example uses
`execution_tier="local"` to demonstrate code execution with policy-based safety controls.
For production with untrusted code, use `execution_tier="docker"`. See file docstring for guidance.**

**Key Features:**

- Accept user input specifying what data to extract and how to visualize it
- Read and process CSV files with dynamic column detection
- Generate Python code using Mellea based on natural language requests
- Execute generated code in an isolated subprocess
- Create headless matplotlib graphs saved to files
- Command-line options for custom CSV files and interactive mode
- 100-row sample dataset with work location information

**Prerequisites:**

Before running this example, ensure you have:

1. Ollama running with the granite4.1:3b model:
   ```bash
   ollama serve  # in one terminal
   ollama pull granite4.1:3b  # in another terminal (if not already downloaded)
   ```

2. Required dependencies installed:
   ```bash
   uv pip install matplotlib numpy
   ```

**Usage Examples:**

```bash
# Default sample data with predefined requests
uv run python docs/examples/requirements/code_generation_and_execution.py

# Custom CSV file
uv run python docs/examples/requirements/code_generation_and_execution.py \
  --csv /path/to/data.csv

# Interactive mode
uv run python docs/examples/requirements/code_generation_and_execution.py \
  --interactive

# Combined options
uv run python docs/examples/requirements/code_generation_and_execution.py \
  --csv /path/to/data.csv --interactive --output /tmp
```

**Pipeline Steps:**

1. User input specifies data extraction and visualization request
2. CSV file is loaded and previewed
3. Code generation creates Python code using LLM
4. Code extraction parses code from markdown blocks
5. Code execution runs in subprocess with output capture
6. Graph saved as PNG using headless matplotlib (Agg backend)

### matplotlib_plotting.py

Demonstrates how to use matplotlib-specific requirements to validate code that
generates plots.

**Key Features:**

- Validating headless backend configuration (Agg, Cairo, pdf, etc.)
- Ensuring plots are explicitly saved to files
- Checking that required dependencies (matplotlib, numpy) are available
- Multiple code patterns: `plt.savefig()`, `fig.savefig()`
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

The `MatplotlibHeadlessBackend` requirement validates that matplotlib code uses
a headless backend suitable for server environments.

### Using MatplotlibHeadlessBackend

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
- `module://gr.matplotlib.backend_gr` — GR graphics library

## PlotFileSaved

The `PlotFileSaved` requirement validates that plots are explicitly saved to a
specific file path.

### Using PlotFileSaved

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
- Keyword arguments: `fig.savefig(fname='/tmp/plot.png', dpi=300)`

## PlotDependenciesAvailable

The `PlotDependenciesAvailable` requirement validates that matplotlib and numpy
are importable.

### Using PlotDependenciesAvailable

```python
from mellea.stdlib.requirements.plotting import PlotDependenciesAvailable

req = PlotDependenciesAvailable()
result = req.validation_fn(context)
print(result.as_bool())  # True if matplotlib and numpy are available
```

## Execution Tiers & Safety Model

### Available Execution Tiers

When executing generated code via `PythonExecutionReq`, you can choose an execution tier that
balances safety, isolation, and performance:

| Tier | Isolation | Speed | Use Case |
|------|-----------|-------|----------|
| `"static"` | Analysis only (no execution) | Fastest | Fast validation before execution |
| `"local_unsafe"` | None (direct subprocess) | Fast | Trusted code only, max performance |
| `"local"` | None (policy control only) | Fast | Trusted code with resource limits (timeout, stdout/stderr limits) |
| `"docker_unsafe"` | Container isolation | Slower | Untrusted code, no policy restrictions |
| `"docker"` | Container isolation + policy | Slower | RECOMMENDED: Untrusted/LLM code with resource limits |

### When to Use Each Tier

**Use `"local"` or `"local_unsafe"`:**
- Code source is trusted (internal development)
- Running in already-sandboxed environment (CI/CD, Docker container)
- Performance is critical and isolation overhead is unacceptable
- Development/demonstration purposes

**Use `"docker"` (RECOMMENDED for production):**
- Code is LLM-generated or from untrusted sources
- Running in shared environments or production
- Security is more important than speed
- Need container-level process isolation and resource limits

### Safety Example

For production with LLM-generated code:

```python
from mellea.stdlib.requirements.python_reqs import PythonExecutionReq
from mellea.stdlib.tools.execution_policy import CapabilityPolicy

# Safe: Container isolation + timeout control
# Note: Most policy fields are declared only (not enforced at runtime).
# The docker tier enforces: timeout, resource limits, and artifact export paths.
# Check CapabilityPolicy.ENFORCED_* class attributes for enforcement status.
req = PythonExecutionReq(
    execution_tier="docker",  # Container-isolated execution
    policy=CapabilityPolicy(
        timeout=30,           # Kill after 30 seconds (enforced)
    ),
    max_output_chars=10_000,  # Limit output size (enforced)
)
```

## Code Generation and Execution

The `code_generation_and_execution.py` example demonstrates end-to-end code generation
with execution:

### How Code Execution Works

When `m.instruct()` is called with `PythonExecutionReq` (with
`execution_tier="local"`), the requirement validation automatically
executes the generated code during validation. This happens as a **side effect**
of the validation process — before the code is returned to the caller.

`PythonExecutionReq` executes code via an `ExecutionEnvironment` built from the
`execution_tier` and `CapabilityPolicy` you pass — that's what enforces the
timeout and tier behavior. See `mellea/stdlib/requirements/python_reqs.py` for
the full implementation.

**Key point:** In `code_generation_and_execution.py`, after `m.instruct()`
returns, the code has *already been executed*. To verify success, check
if the output file exists (e.g., the PNG graph file). There is no need
for a separate execution step.

### Command-line Options

```text
--csv CSV              Path to CSV file
--interactive          Interactive mode: accept user input
--output OUTPUT        Directory to save generated graphs
--help                 Show help message and usage examples
```

### Sample Dataset Columns

- `name` — Employee name
- `age` — Employee age
- `salary` — Annual salary
- `department` — Department (Engineering, Sales, HR, Marketing, Finance)
- `years_experience` — Years at company
- `work_location` — Office location (10 cities)

### Example Requests

```text
Extract average salary by department and create a bar chart
Extract employee count by work location and create a bar chart
Extract age and salary for all employees and create a scatter plot
Extract salary by years of experience and create a line plot
Extract employee distribution across work locations and create a pie chart
```

## Use Cases

**When to Use:**

- Generating data visualization code from natural language
- Extracting specific data from CSV files with LLM assistance
- Creating reproducible data analysis pipelines
- Generating plotting code that must run on servers (headless)
- Validating plots are saved to files (not displayed interactively)
- Ensuring code can run in CI/CD environments
- Verifying required data science libraries are available
- Code quality gates for machine learning notebooks

**When NOT to Use:**

- Interactive plotting applications
- Jupyter notebooks meant for display (use `nbAgg` backend instead)
- Desktop applications with display servers

## Related Documentation

- See `mellea/stdlib/requirements/plotting/matplotlib.py` for implementation
- See `test/stdlib/requirements/plotting/test_matplotlib.py` for tests

## Requirements

### matplotlib_plotting.py

- matplotlib and numpy installed (auto-checked by requirements)
- Any code analysis backend (requirements use AST analysis)

### code_generation_and_execution.py

- Ollama backend with granite4.1:3b model running (`ollama serve`)
- matplotlib and numpy installed (required for executing generated code)
- Python subprocess execution support (provided by `PythonExecutionReq`)
