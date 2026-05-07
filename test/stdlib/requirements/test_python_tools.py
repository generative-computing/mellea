"""Tests for Python tool requirements bundle."""

import tempfile
from pathlib import Path

from mellea.core import Context, ModelOutputThunk
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.python_tools import (
    PythonToolRequirements,
    _calls_savefig,
    _code_parses,
    _get_unauthorized_imports,
    _sets_headless_backend,
    _uses_pyplot_plot,
    _uses_pyplot_show,
)


def from_model(content: str) -> Context:
    """Helper to create context from model output."""
    ctx = ChatContext()
    ctx = ctx.add(ModelOutputThunk(value=content))
    return ctx


# region: Helper function tests


class TestCodeParses:
    """Tests for _code_parses helper."""

    def test_valid_code_parses(self):
        """Valid Python code should parse."""
        code = "x = 1\nprint(x)"
        parses, error = _code_parses(code)
        assert parses is True
        assert error is None

    def test_syntax_error_detected(self):
        """Syntax errors should be detected."""
        code = "def foo(\n    return 42"
        parses, error = _code_parses(code)
        assert parses is False
        assert error is not None
        assert "Syntax error" in error

    def test_missing_colon(self):
        """Missing colons should be detected."""
        code = "if True\n    print('hello')"
        parses, error = _code_parses(code)
        assert parses is False
        assert error is not None


class TestUnauthorizedImports:
    """Tests for _get_unauthorized_imports helper."""

    def test_no_imports_allowed_always(self):
        """Code with no imports should always pass."""
        code = "x = 1"
        unauthorized = _get_unauthorized_imports(code, ["numpy"])
        assert unauthorized == []

    def test_allowed_imports(self):
        """Allowed imports should not be flagged."""
        code = "import numpy\nimport pandas"
        unauthorized = _get_unauthorized_imports(code, ["numpy", "pandas"])
        assert unauthorized == []

    def test_unauthorized_import_detected(self):
        """Unauthorized imports should be detected."""
        code = "import subprocess"
        unauthorized = _get_unauthorized_imports(code, ["numpy", "pandas"])
        assert "subprocess" in unauthorized

    def test_none_allows_all(self):
        """allowed_imports=None should allow any import."""
        code = "import subprocess\nimport socket\nimport os"
        unauthorized = _get_unauthorized_imports(code, None)
        assert unauthorized == []

    def test_nested_imports(self):
        """Only top-level module checked for nested imports."""
        code = "import numpy.random"
        unauthorized = _get_unauthorized_imports(code, ["numpy"])
        assert unauthorized == []

    def test_from_import(self):
        """from ... import should be checked."""
        code = "from matplotlib import pyplot"
        unauthorized = _get_unauthorized_imports(code, ["numpy"])
        assert "matplotlib" in unauthorized

    def test_multiple_unauthorized(self):
        """Multiple unauthorized imports should be detected."""
        code = "import subprocess\nimport socket"
        unauthorized = _get_unauthorized_imports(code, ["numpy"])
        assert len(unauthorized) == 2
        assert "subprocess" in unauthorized
        assert "socket" in unauthorized


class TestMatplotlibDetection:
    """Tests for matplotlib-related detection functions."""

    def test_plt_show_detected(self):
        """plt.show() should be detected."""
        code = "plt.plot([1, 2, 3])\nplt.show()"
        assert _uses_pyplot_show(code) is True

    def test_plt_show_not_detected(self):
        """Code without plt.show() should pass."""
        code = "plt.plot([1, 2, 3])\nplt.savefig('plot.png')"
        assert _uses_pyplot_show(code) is False

    def test_headless_backend_agg(self):
        """Agg backend should be detected."""
        code = "import matplotlib\nmatplotlib.use('Agg')"
        assert _sets_headless_backend(code) is True

    def test_headless_backend_variations(self):
        """Various headless backends should be detected."""
        for backend in ["Agg", "Svg", "Cairo", "PDF", "PS"]:
            code = f"matplotlib.use('{backend}')"
            assert _sets_headless_backend(code) is True

    def test_non_headless_backend(self):
        """Non-headless backends should not be detected."""
        code = "matplotlib.use('TkAgg')"
        assert _sets_headless_backend(code) is False

    def test_uses_pyplot_plot(self):
        """Plotting functions should be detected."""
        for func in ["plt.plot", "plt.bar", "plt.scatter", ".plot("]:
            code = f"import matplotlib.pyplot as plt\n{func}([1, 2, 3])"
            assert _uses_pyplot_plot(code) is True

    def test_savefig_detected(self):
        """savefig call should be detected."""
        code = "plt.plot([1, 2, 3])\nplt.savefig('plot.png')"
        assert _calls_savefig(code) is True

    def test_savefig_not_detected(self):
        """Code without savefig should not be detected."""
        code = "plt.plot([1, 2, 3])\nplt.show()"
        assert _calls_savefig(code) is False


# endregion


# region: PythonToolRequirements tests


class TestPythonToolRequirementsBasic:
    """Basic tests for PythonToolRequirements bundle."""

    def test_initialization(self):
        """Bundle should initialize with default settings."""
        bundle = PythonToolRequirements()
        assert bundle.requirements is not None
        assert len(bundle.requirements) > 0

    def test_with_output_path(self):
        """Bundle should accept output_path parameter."""
        bundle = PythonToolRequirements(output_path="/tmp/plot.png")
        assert bundle.output_path == "/tmp/plot.png"

    def test_with_allowed_imports(self):
        """Bundle should accept allowed_imports parameter."""
        allowed = ["numpy", "matplotlib"]
        bundle = PythonToolRequirements(allowed_imports=allowed)
        assert bundle.allowed_imports == allowed

    def test_output_artifact_checking_enabled_by_default(self):
        """Output artifact checking should be enabled if output_path is set."""
        bundle = PythonToolRequirements(output_path="/tmp/plot.png")
        assert bundle._check_output_artifacts is True

    def test_output_artifact_checking_disabled_by_default(self):
        """Output artifact checking should be disabled if no output_path."""
        bundle = PythonToolRequirements()
        assert bundle._check_output_artifacts is False

    def test_repr(self):
        """Bundle should have a readable repr."""
        bundle = PythonToolRequirements(output_path="/tmp/plot.png")
        repr_str = repr(bundle)
        assert "PythonToolRequirements" in repr_str
        assert "/tmp/plot.png" in repr_str


# endregion


# region: Individual requirement validation tests


class TestMustInvokePythonTool:
    """Tests for MustInvokePythonTool requirement."""

    def test_tool_not_called(self):
        """Should fail if python tool not called."""
        bundle = PythonToolRequirements()
        req = bundle.requirements[0]

        ctx = from_model("Here is the code:\n```python\nprint('hello')\n```")
        result = req.validation_fn(ctx)

        assert result.as_bool() is False
        reason_lower = result.reason.lower()
        assert "did not invoke" in reason_lower or "did not call" in reason_lower

    def test_python_tool_called(self):
        """Should pass if python tool is called."""
        bundle = PythonToolRequirements()
        req = bundle.requirements[0]

        ctx = ChatContext()
        call_obj = type("Call", (), {"args": {"code": "print('hi')"}})()
        output = ModelOutputThunk(
            value="I'll execute this code", tool_calls={"python": call_obj}
        )
        ctx = ctx.add(output)

        result = req.validation_fn(ctx)
        assert result.as_bool() is True


class TestPythonToolHasCodeArg:
    """Tests for PythonToolHasCodeArg requirement."""

    def test_missing_code_argument(self):
        """Should fail if python tool call has no code argument."""
        bundle = PythonToolRequirements()
        req = bundle.requirements[1]

        ctx = ChatContext()
        call_obj = type("Call", (), {"args": {"other": "value"}})()
        output = ModelOutputThunk(
            value="I'll execute this", tool_calls={"python": call_obj}
        )
        ctx = ctx.add(output)

        result = req.validation_fn(ctx)
        assert result.as_bool() is False
        assert "code" in result.reason.lower()

    def test_has_code_argument(self):
        """Should pass if python tool call has code argument."""
        bundle = PythonToolRequirements()
        req = bundle.requirements[1]

        ctx = ChatContext()
        call_obj = type("Call", (), {"args": {"code": "print('hi')"}})()
        output = ModelOutputThunk(
            value="I'll execute this", tool_calls={"python": call_obj}
        )
        ctx = ctx.add(output)

        result = req.validation_fn(ctx)
        assert result.as_bool() is True


class TestCodeParsesRequirement:
    """Tests for code parsing requirement."""

    def test_valid_code(self):
        """Valid code should pass."""
        bundle = PythonToolRequirements()
        parse_reqs = [
            r for r in bundle.requirements if "parse" in r.description.lower()
        ]
        parse_req = parse_reqs[0]

        ctx = from_model("```python\nx = 1\nprint(x)\n```")
        result = parse_req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_syntax_error(self):
        """Syntax errors should be caught."""
        bundle = PythonToolRequirements()
        parse_reqs = [
            r for r in bundle.requirements if "parse" in r.description.lower()
        ]
        parse_req = parse_reqs[0]

        ctx = from_model("```python\ndef foo(\n    return 42\n```")
        result = parse_req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "syntax" in result.reason.lower()

    def test_valid_code_from_tool_calls(self):
        """Valid code in tool_calls should parse."""
        bundle = PythonToolRequirements()
        parse_reqs = [
            r for r in bundle.requirements if "parse" in r.description.lower()
        ]
        parse_req = parse_reqs[0]

        # Create context with tool_calls instead of markdown
        ctx = ChatContext()
        tool_call = type("Call", (), {"args": {"code": "x = 1\nprint(x)"}})()
        output = ModelOutputThunk(value="", tool_calls={"python": tool_call})
        ctx = ctx.add(output)

        result = parse_req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_syntax_error_from_tool_calls(self):
        """Syntax errors in tool_calls should be caught."""
        bundle = PythonToolRequirements()
        parse_reqs = [
            r for r in bundle.requirements if "parse" in r.description.lower()
        ]
        parse_req = parse_reqs[0]

        # Create context with tool_calls containing syntax error
        ctx = ChatContext()
        tool_call = type("Call", (), {"args": {"code": "def foo(\n    return 42"}})()
        output = ModelOutputThunk(value="", tool_calls={"python": tool_call})
        ctx = ctx.add(output)

        result = parse_req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "syntax" in result.reason.lower()


class TestImportAllowlistRequirement:
    """Tests for import allowlist requirement."""

    def test_allowed_imports(self):
        """Allowed imports should pass."""
        allowed = ["numpy", "matplotlib"]
        bundle = PythonToolRequirements(allowed_imports=allowed)

        import_reqs = [
            r for r in bundle.requirements if "import" in r.description.lower()
        ]
        assert len(import_reqs) > 0
        import_req = import_reqs[0]

        ctx = from_model(
            "```python\nimport numpy\nimport matplotlib.pyplot as plt\n```"
        )
        result = import_req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_unauthorized_imports(self):
        """Unauthorized imports should fail."""
        allowed = ["numpy"]
        bundle = PythonToolRequirements(allowed_imports=allowed)

        import_reqs = [
            r for r in bundle.requirements if "import" in r.description.lower()
        ]
        import_req = import_reqs[0]

        ctx = from_model("```python\nimport subprocess\n```")
        result = import_req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "subprocess" in result.reason

    def test_allowed_imports_from_tool_calls(self):
        """Allowed imports in tool_calls should pass."""
        allowed = ["numpy", "matplotlib"]
        bundle = PythonToolRequirements(allowed_imports=allowed)

        import_reqs = [
            r for r in bundle.requirements if "import" in r.description.lower()
        ]
        import_req = import_reqs[0]

        # Create context with tool_calls
        ctx = ChatContext()
        code = "import numpy\nimport matplotlib.pyplot as plt\nprint(numpy.pi)"
        tool_call = type("Call", (), {"args": {"code": code}})()
        output = ModelOutputThunk(value="", tool_calls={"python": tool_call})
        ctx = ctx.add(output)

        result = import_req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_unauthorized_imports_from_tool_calls(self):
        """Unauthorized imports in tool_calls should fail."""
        allowed = ["numpy"]
        bundle = PythonToolRequirements(allowed_imports=allowed)

        import_reqs = [
            r for r in bundle.requirements if "import" in r.description.lower()
        ]
        import_req = import_reqs[0]

        # Create context with tool_calls
        ctx = ChatContext()
        tool_call = type("Call", (), {"args": {"code": "import subprocess\n"}})()
        output = ModelOutputThunk(value="", tool_calls={"python": tool_call})
        ctx = ctx.add(output)

        result = import_req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "subprocess" in result.reason


class TestMatplotlibHeadlessRequirement:
    """Tests for matplotlib headless backend requirement."""

    def test_plt_show_without_backend(self):
        """plt.show() without headless backend should fail."""
        bundle = PythonToolRequirements()
        matplotlib_reqs = [
            r for r in bundle.requirements if "headless" in r.description.lower()
        ]
        assert len(matplotlib_reqs) > 0
        matplotlib_req = matplotlib_reqs[0]

        ctx = from_model("```python\nplt.plot([1, 2, 3])\nplt.show()\n```")
        result = matplotlib_req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "headless" in result.reason.lower() or "Agg" in result.reason

    def test_plt_show_with_backend(self):
        """plt.show() with headless backend should pass."""
        bundle = PythonToolRequirements()
        matplotlib_reqs = [
            r for r in bundle.requirements if "headless" in r.description.lower()
        ]
        matplotlib_req = matplotlib_reqs[0]

        ctx = from_model(
            "```python\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "plt.plot([1, 2, 3])\n"
            "plt.show()\n"
            "```"
        )
        result = matplotlib_req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_no_plt_show(self):
        """Code without plt.show() should pass."""
        bundle = PythonToolRequirements()
        matplotlib_reqs = [
            r for r in bundle.requirements if "headless" in r.description.lower()
        ]
        matplotlib_req = matplotlib_reqs[0]

        ctx = from_model("```python\nplt.plot([1, 2, 3])\nplt.savefig('plot.png')\n```")
        result = matplotlib_req.validation_fn(ctx)

        assert result.as_bool() is True


class TestPlotsAreSavedRequirement:
    """Tests for plots must be saved requirement."""

    def test_plot_without_savefig(self):
        """Plotting without savefig should fail."""
        bundle = PythonToolRequirements()
        plot_reqs = [
            r for r in bundle.requirements if "savefig" in r.description.lower()
        ]
        assert len(plot_reqs) > 0
        plot_req = plot_reqs[0]

        ctx = from_model("```python\nplt.plot([1, 2, 3])\nplt.show()\n```")
        result = plot_req.validation_fn(ctx)

        assert result.as_bool() is False
        assert "savefig" in result.reason.lower()

    def test_plot_with_savefig(self):
        """Plotting with savefig should pass."""
        bundle = PythonToolRequirements()
        plot_reqs = [
            r for r in bundle.requirements if "savefig" in r.description.lower()
        ]
        plot_req = plot_reqs[0]

        ctx = from_model("```python\nplt.plot([1, 2, 3])\nplt.savefig('plot.png')\n```")
        result = plot_req.validation_fn(ctx)

        assert result.as_bool() is True

    def test_no_plotting(self):
        """Code without plotting should pass."""
        bundle = PythonToolRequirements()
        plot_reqs = [
            r for r in bundle.requirements if "savefig" in r.description.lower()
        ]
        plot_req = plot_reqs[0]

        ctx = from_model("```python\nx = 1\nprint(x)\n```")
        result = plot_req.validation_fn(ctx)

        assert result.as_bool() is True


class TestOutputArtifactsRequirement:
    """Tests for output artifacts requirement."""

    def test_output_file_not_created(self):
        """Should fail if output file not created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "plot.png")

            bundle = PythonToolRequirements(output_path=output_path)
            artifact_reqs = [
                r for r in bundle.requirements if "output file" in r.description.lower()
            ]
            assert len(artifact_reqs) > 0
            artifact_req = artifact_reqs[0]

            ctx = from_model("Code ran successfully")
            result = artifact_req.validation_fn(ctx)

            assert result.as_bool() is False
            assert output_path in result.reason

    def test_output_file_exists_and_nonempty(self):
        """Should pass if output file exists and is non-empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "plot.png")
            Path(output_path).write_bytes(b"fake png data")

            bundle = PythonToolRequirements(output_path=output_path)
            artifact_reqs = [
                r for r in bundle.requirements if "output file" in r.description.lower()
            ]
            artifact_req = artifact_reqs[0]

            ctx = from_model("Code ran successfully")
            result = artifact_req.validation_fn(ctx)

            assert result.as_bool() is True

    def test_output_file_empty(self):
        """Should fail if output file exists but is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "plot.png")
            Path(output_path).write_bytes(b"")

            bundle = PythonToolRequirements(output_path=output_path)
            artifact_reqs = [
                r for r in bundle.requirements if "output file" in r.description.lower()
            ]
            artifact_req = artifact_reqs[0]

            ctx = from_model("Code ran successfully")
            result = artifact_req.validation_fn(ctx)

            assert result.as_bool() is False
            assert "empty" in result.reason.lower()

    def test_output_artifact_disabled_without_output_path(self):
        """Output artifact requirement should not be present without output_path."""
        bundle = PythonToolRequirements()
        artifact_reqs = [
            r for r in bundle.requirements if "output file" in r.description.lower()
        ]
        assert len(artifact_reqs) == 0


# endregion
