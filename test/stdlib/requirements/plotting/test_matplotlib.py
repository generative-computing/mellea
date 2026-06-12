"""Tests for matplotlib-specific requirements."""

import pytest

from mellea.core import Context, ModelOutputThunk, Requirement
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.plotting import (
    MatplotlibHeadlessBackend,
    PlotDependenciesAvailable,
    PlotFileSaved,
    python_plotting_requirements,
)
from mellea.stdlib.requirements.python_tools import (
    ImportRestrictions,
    NoImportRestrictions,
    PythonCodeExtraction,
    PythonExecutionReq,
    PythonSyntaxValid,
)


def from_model(content: str) -> Context:
    """Helper to create context from model output."""
    ctx = ChatContext()
    ctx = ctx.add(ModelOutputThunk(value=content))
    return ctx


# Test fixtures for MatplotlibHeadlessBackend

CODE_WITH_HEADLESS_AGG = """```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 2, 3])
plt.show()
```"""

CODE_WITH_HEADLESS_CAIRO = """```python
import matplotlib
matplotlib.use('Cairo')
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 2, 3])
```"""

CODE_WITH_HEADLESS_PDF = """```python
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
```"""

CODE_WITH_HEADLESS_AGG_KEYWORD = """```python
import matplotlib
matplotlib.use(backend='Agg')
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 2, 3])
```"""

CODE_WITH_INTERACTIVE_TKAGG = """```python
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
```"""

CODE_WITHOUT_MATPLOTLIB_USE = """```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
plt.show()
```"""

CODE_WITH_MATPLOTLIB_SHOW_ONLY = """```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 4, 9]
plt.plot(x, y)
plt.show()
```"""

# Test fixtures for PlotFileSaved

CODE_WITH_SAVEFIG_PLT = """```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 2, 3])
plt.savefig('/tmp/plot.png')
```"""

CODE_WITH_SAVEFIG_FIG = """```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 2, 3])
fig.savefig('/tmp/plot.png')
```"""

CODE_WITH_SAVEFIG_KEYWORD = """```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3])
fig.savefig(fname='/tmp/plot.png')
```"""

CODE_WITH_SAVEFIG_WRONG_PATH = """```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
plt.savefig('/tmp/other_plot.png')
```"""

CODE_WITH_SHOW_NO_SAVEFIG = """```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
plt.show()
```"""

CODE_WITHOUT_SAVEFIG = """```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
print("Plot created")
```"""

CODE_WITH_MULTIPLE_SAVEFIG_CALLS = """```python
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()
ax1.plot([1, 2, 3])
fig1.savefig('/tmp/plot.png')

fig2, ax2 = plt.subplots()
ax2.plot([4, 5, 6])
fig2.savefig('/tmp/other.png')
```"""

# Test fixtures for PlotDependenciesAvailable

CODE_WITH_MATPLOTLIB_IMPORT = """```python
import matplotlib
import numpy as np

data = np.array([1, 2, 3])
print(data)
```"""

CODE_WITH_MATPLOTLIB_PYPLOT_IMPORT = """```python
import matplotlib.pyplot as plt
import numpy

plt.plot([1, 2, 3])
```"""


class TestMatplotlibHeadlessBackend:
    """Tests for MatplotlibHeadlessBackend requirement."""

    def test_valid_agg_backend(self):
        """Test pass with Agg backend."""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(CODE_WITH_HEADLESS_AGG)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "Agg" in (result.reason or "")

    def test_valid_cairo_backend(self):
        """Test pass with Cairo backend."""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(CODE_WITH_HEADLESS_CAIRO)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "Cairo" in (result.reason or "")

    def test_valid_pdf_backend(self):
        """Test pass with PDF backend."""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(CODE_WITH_HEADLESS_PDF)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "pdf" in (result.reason or "")

    def test_valid_agg_backend_as_keyword(self):
        """Test pass with Agg backend specified as keyword argument."""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(CODE_WITH_HEADLESS_AGG_KEYWORD)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "Agg" in (result.reason or "")

    def test_invalid_interactive_tkagg_backend(self):
        """Test fail with interactive TkAgg backend."""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(CODE_WITH_INTERACTIVE_TKAGG)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "not headless" in (result.reason or "").lower()

    def test_missing_matplotlib_use_call(self):
        """Test fail when matplotlib.use() is missing."""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(CODE_WITHOUT_MATPLOTLIB_USE)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "matplotlib.use()" in (result.reason or "")

    def test_no_code_blocks(self):
        """Test fail when no code blocks present."""
        req = MatplotlibHeadlessBackend()
        ctx = from_model("Just some text about matplotlib, no code.")
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "could not extract" in (result.reason or "").lower()

    def test_requirement_description(self):
        """Test requirement has appropriate description."""
        req = MatplotlibHeadlessBackend()
        assert "headless" in req.description.lower()
        assert "matplotlib" in req.description.lower()


class TestPlotFileSaved:
    """Tests for PlotFileSaved requirement."""

    def test_valid_plt_savefig(self):
        """Test pass with plt.savefig()."""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model(CODE_WITH_SAVEFIG_PLT)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "/tmp/plot.png" in (result.reason or "")

    def test_valid_fig_savefig(self):
        """Test pass with fig.savefig()."""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model(CODE_WITH_SAVEFIG_FIG)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "/tmp/plot.png" in (result.reason or "")

    def test_valid_savefig_keyword_argument(self):
        """Test pass with savefig(fname=...) keyword argument."""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model(CODE_WITH_SAVEFIG_KEYWORD)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "/tmp/plot.png" in (result.reason or "")

    def test_invalid_wrong_output_path(self):
        """Test fail when savefig() uses wrong path."""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model(CODE_WITH_SAVEFIG_WRONG_PATH)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "/tmp/plot.png" in (result.reason or "")

    def test_invalid_show_but_no_savefig(self):
        """Test fail when plt.show() used but no savefig()."""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model(CODE_WITH_SHOW_NO_SAVEFIG)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "savefig()" in (result.reason or "").lower()

    def test_invalid_no_savefig_call(self):
        """Test fail when savefig() is not called at all."""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model(CODE_WITHOUT_SAVEFIG)
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "savefig()" in (result.reason or "").lower()

    def test_pass_with_multiple_savefig_calls(self):
        """Test pass when target path appears in any savefig() call."""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model(CODE_WITH_MULTIPLE_SAVEFIG_CALLS)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "/tmp/plot.png" in (result.reason or "")

    def test_no_code_blocks(self):
        """Test fail when no code blocks present."""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model("No code here, just text about plotting.")
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "could not extract" in (result.reason or "").lower()

    def test_requirement_description_includes_output_path(self):
        """Test requirement description includes the output_path."""
        output_path = "/output/my_plot.png"
        req = PlotFileSaved(output_path=output_path)
        assert output_path in req.description

    def test_output_path_stored(self):
        """Test that output_path is stored on the requirement."""
        output_path = "/tmp/plot.png"
        req = PlotFileSaved(output_path=output_path)
        assert req.output_path == output_path


class TestPlotDependenciesAvailable:
    """Tests for PlotDependenciesAvailable requirement."""

    @pytest.mark.xfail(reason="matplotlib not available in test environment")
    def test_valid_dependencies_available(self):
        """Test pass when matplotlib and numpy are available."""
        req = PlotDependenciesAvailable()
        # Context is unused since requirement checks environment, not code
        ctx = from_model(CODE_WITH_MATPLOTLIB_IMPORT)
        result = req.validation_fn(ctx)

        assert result.as_bool() is True
        assert "dependencies available" in (result.reason or "").lower()

    def test_requirement_description(self):
        """Test requirement has appropriate description."""
        req = PlotDependenciesAvailable()
        assert "matplotlib" in req.description.lower()
        assert "numpy" in req.description.lower()


class TestMatplotlibHeadlessBackendEdgeCases:
    """Edge case tests for MatplotlibHeadlessBackend."""

    def test_backend_case_insensitive(self):
        """Test that backend names are matched case-insensitively, consistent with matplotlib's runtime behaviour.

        Matplotlib normalises backend names at runtime — 'agg' and 'Agg' select the same backend.
        """
        code_lowercase = """```python
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
```"""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(code_lowercase)
        result = req.validation_fn(ctx)
        assert result.as_bool() is True
        assert "agg" in (result.reason or "").lower()

    def test_cairo_lowercase_backend(self):
        """Test that lowercase 'cairo' is accepted and mapped to canonical 'Cairo'."""
        code_lowercase = """```python
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
```"""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(code_lowercase)
        result = req.validation_fn(ctx)
        assert result.as_bool() is True
        assert "Cairo" in (result.reason or "")

    def test_svg_lowercase_backend(self):
        """Test that lowercase 'svg' is accepted."""
        code_lowercase = """```python
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
```"""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(code_lowercase)
        result = req.validation_fn(ctx)
        assert result.as_bool() is True
        assert "svg" in (result.reason or "").lower()

    def test_matplotlib_use_with_variable(self):
        """Test that matplotlib.use() with variable backend is not detected."""
        code_with_var = """```python
import matplotlib
backend = 'Agg'
matplotlib.use(backend)
```"""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(code_with_var)
        result = req.validation_fn(ctx)
        # Variable arguments are not detected, so this should fail
        assert result.as_bool() is False
        assert "non-literal" in (result.reason or "").lower()

    def test_multiple_matplotlib_use_calls(self):
        """Test that the validator uses the first matplotlib.use() call with a literal argument, regardless of whether it is headless."""
        code_multi = """```python
import matplotlib
matplotlib.use('Agg')
# later attempt to switch
matplotlib.use('TkAgg')
```"""
        req = MatplotlibHeadlessBackend()
        ctx = from_model(code_multi)
        result = req.validation_fn(ctx)
        # Passes: 'Agg' is the first literal use() arg found — note that at runtime matplotlib would apply 'TkAgg' (the last call)
        assert result.as_bool() is True


class TestPlotFileSavedEdgeCases:
    """Edge case tests for PlotFileSaved."""

    def test_savefig_with_variable_path(self):
        """Test that savefig() with variable path is not detected."""
        code_with_var = """```python
import matplotlib.pyplot as plt
output_path = '/tmp/plot.png'
plt.savefig(output_path)
```"""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model(code_with_var)
        result = req.validation_fn(ctx)
        # Variable arguments are not detected, so this should fail
        assert result.as_bool() is False

    def test_different_output_extensions(self):
        """Test that exact path matching is required."""
        code_jpg = """```python
import matplotlib.pyplot as plt
plt.savefig('/tmp/plot.jpg')
```"""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model(code_jpg)
        result = req.validation_fn(ctx)
        # Paths don't match, so should fail
        assert result.as_bool() is False

    def test_path_as_keyword_with_different_names(self):
        """Test different keyword argument names for savefig."""
        code_filename = """```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.savefig(fname='/tmp/plot.png')
```"""
        req = PlotFileSaved(output_path="/tmp/plot.png")
        ctx = from_model(code_filename)
        result = req.validation_fn(ctx)
        # fname= is the correct keyword; savefig has no 'filename' parameter
        assert result.as_bool() is True


class TestPythonPlottingRequirementsFactory:
    """Tests for python_plotting_requirements factory function."""

    def test_factory_returns_list_of_requirements(self):
        """Test that factory returns a list of seven Requirement instances."""
        output_path = "/tmp/plot.png"
        reqs = python_plotting_requirements(output_path=output_path)

        assert isinstance(reqs, list)
        assert len(reqs) == 7
        assert all(isinstance(r, Requirement) for r in reqs)

    def test_factory_returns_correct_requirement_order(self):
        """Test requirements are in expected order (python tools + plotting)."""
        output_path = "/tmp/plot.png"
        reqs = python_plotting_requirements(output_path=output_path)

        # First 4 from python_code_generation_requirements
        assert isinstance(reqs[0], PythonCodeExtraction)
        assert isinstance(reqs[1], PythonSyntaxValid)
        assert isinstance(reqs[2], PythonExecutionReq)
        assert isinstance(
            reqs[3], NoImportRestrictions
        )  # No allowed_imports, so NoImportRestrictions

        # Last 3 plotting-specific
        assert isinstance(reqs[4], MatplotlibHeadlessBackend)
        assert isinstance(reqs[5], PlotFileSaved)
        assert isinstance(reqs[6], PlotDependenciesAvailable)

    def test_factory_with_allowed_imports(self):
        """Test that allowed_imports parameter creates ImportRestrictions."""
        output_path = "/tmp/plot.png"
        allowed_imports = ["matplotlib", "numpy"]
        reqs = python_plotting_requirements(
            output_path=output_path, allowed_imports=allowed_imports
        )

        assert len(reqs) == 7
        assert isinstance(reqs[3], ImportRestrictions)
        assert reqs[3].allowed_imports == allowed_imports

    def test_factory_propagates_output_path(self):
        """Test that output_path is correctly passed to PlotFileSaved."""
        output_path = "/output/my_plot.png"
        reqs = python_plotting_requirements(output_path=output_path)

        plot_saved_req = reqs[5]
        assert isinstance(plot_saved_req, PlotFileSaved)
        assert plot_saved_req.output_path == output_path
        assert output_path in plot_saved_req.description

    def test_factory_with_different_paths(self):
        """Test factory works with various output path formats."""
        test_paths = [
            "/tmp/plot.png",
            "/output/figures/chart.svg",
            "plot.pdf",
            "/var/tmp/matplotlib_output.jpg",
        ]

        for path in test_paths:
            reqs = python_plotting_requirements(output_path=path)
            assert len(reqs) == 7
            assert reqs[5].output_path == path

    def test_factory_raises_on_empty_path(self):
        """Test that factory raises ValueError for empty output_path."""
        with pytest.raises(ValueError):
            python_plotting_requirements(output_path="")

        with pytest.raises(ValueError):
            python_plotting_requirements(output_path="   ")

    def test_factory_propagates_timeout_seconds_validation(self):
        """Test that invalid timeout_seconds from delegated factory propagates."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            python_plotting_requirements(output_path="/tmp/plot.png", timeout_seconds=0)

    def test_factory_propagates_output_limit_chars_validation(self):
        """Test that invalid output_limit_chars from delegated factory propagates."""
        with pytest.raises(ValueError, match="output_limit_chars must be positive"):
            python_plotting_requirements(
                output_path="/tmp/plot.png", output_limit_chars=-1
            )

    def test_factory_can_be_unpacked(self):
        """Test that factory result can be unpacked into all 7 requirements."""
        output_path = "/tmp/plot.png"
        r1, r2, r3, r4, r5, r6, r7 = python_plotting_requirements(
            output_path=output_path
        )
        assert isinstance(r1, PythonCodeExtraction)
        assert isinstance(r2, PythonSyntaxValid)
        assert isinstance(r3, PythonExecutionReq)
        assert isinstance(r4, NoImportRestrictions)
        assert isinstance(r5, MatplotlibHeadlessBackend)
        assert isinstance(r6, PlotFileSaved)
        assert isinstance(r7, PlotDependenciesAvailable)

    def test_factory_requirements_are_independent_instances(self):
        """Test that multiple factory calls create independent requirement instances."""
        reqs1 = python_plotting_requirements(output_path="/tmp/plot1.png")
        reqs2 = python_plotting_requirements(output_path="/tmp/plot2.png")

        # Different instances
        assert reqs1[5] is not reqs2[5]
        # But different output paths
        assert reqs1[5].output_path != reqs2[5].output_path

    def test_factory_with_custom_execution_parameters(self):
        """Test that custom execution parameters can be passed through."""
        output_path = "/tmp/plot.png"
        reqs = python_plotting_requirements(
            output_path=output_path,
            output_limit_chars=5000,
            timeout_seconds=10,
            use_sandbox=True,
        )

        assert len(reqs) == 7
        execution_req = reqs[2]
        assert isinstance(execution_req, PythonExecutionReq)
