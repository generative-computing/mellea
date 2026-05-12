"""Tests for Python tool requirements bundle."""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from mellea.core import (
    Context,
    ModelOutputThunk,
    ModelToolCall,
    Requirement,
    ValidationResult,
)
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.plotting.matplotlib import (
    _calls_savefig,
    _sets_headless_backend,
    _uses_pyplot_plot,
    _uses_pyplot_show,
)
from mellea.stdlib.requirements.python_tools import (
    _code_parses,
    python_tool_requirements,
)
from mellea.stdlib.tools.interpreter import get_unauthorized_imports


def from_model(content: str) -> Context:
    """Helper to create context from model output."""
    ctx = ChatContext()
    ctx = ctx.add(ModelOutputThunk(value=content))
    return ctx


def requirement_description(requirement: Requirement) -> str:
    """Return a non-optional description for test filtering."""
    assert requirement.description is not None
    return requirement.description


def validation_fn(requirement: Requirement) -> Callable[[Context], ValidationResult]:
    """Return a requirement validation function for tests."""
    assert requirement.validation_fn is not None
    return requirement.validation_fn


def validation_reason(result: ValidationResult) -> str:
    """Return a non-optional validation reason for assertions."""
    assert result.reason is not None
    return result.reason


class _DummyTool:
    """Minimal tool stub for constructing ModelToolCall in tests."""

    def run(self, **kwargs: Any) -> None:
        return None


def python_tool_call(code: str | None = None, **extra_args: str) -> ModelToolCall:
    """Create a typed python tool call for tests."""
    args: dict[str, str] = dict(extra_args)
    if code is not None:
        args["code"] = code
    return ModelToolCall(name="python", func=cast(Any, _DummyTool()), args=args)


def requirements_matching(
    substring: str,
    *,
    output_path: str | None = None,
    allowed_imports: list[str] | None = None,
    output_limit_bytes: int = 50_000,
    check_output_artifacts: bool | None = None,
) -> list[Requirement]:
    """Return requirements whose description contains the substring."""
    return [
        requirement
        for requirement in python_tool_requirements(
            output_path=output_path,
            allowed_imports=allowed_imports,
            output_limit_bytes=output_limit_bytes,
            check_output_artifacts=check_output_artifacts,
        )
        if substring in requirement_description(requirement).lower()
    ]


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
    """Tests for get_unauthorized_imports helper."""

    def test_no_imports_allowed_always(self):
        """Code with no imports should always pass."""
        code = "x = 1"
        unauthorized = get_unauthorized_imports(code, ["numpy"])
        assert unauthorized == []

    def test_allowed_imports(self):
        """Allowed imports should not be flagged."""
        code = "import numpy\nimport pandas"
        unauthorized = get_unauthorized_imports(code, ["numpy", "pandas"])
        assert unauthorized == []

    def test_unauthorized_import_detected(self):
        """Unauthorized imports should be detected."""
        code = "import subprocess"
        unauthorized = get_unauthorized_imports(code, ["numpy", "pandas"])
        assert "subprocess" in unauthorized

    def test_none_allows_all(self):
        """allowed_imports=None should allow any import."""
        code = "import subprocess\nimport socket\nimport os"
        unauthorized = get_unauthorized_imports(code, None)
        assert unauthorized == []

    def test_nested_imports(self):
        """Only top-level module checked for nested imports."""
        code = "import numpy.random"
        unauthorized = get_unauthorized_imports(code, ["numpy"])
        assert unauthorized == []

    def test_from_import(self):
        """from ... import should be checked."""
        code = "from matplotlib import pyplot"
        unauthorized = get_unauthorized_imports(code, ["numpy"])
        assert "matplotlib" in unauthorized

    def test_multiple_unauthorized(self):
        """Multiple unauthorized imports should be detected."""
        code = "import subprocess\nimport socket"
        unauthorized = get_unauthorized_imports(code, ["numpy"])
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


# region: python_tool_requirements tests


class TestPythonToolRequirementsBasic:
    """Basic tests for python_tool_requirements."""

    def test_initialization(self):
        """Factory should return requirements with default settings."""
        requirements = python_tool_requirements()
        assert requirements
        assert len(requirements) > 0

    def test_with_output_path_enables_artifact_requirement(self):
        """Output path should enable artifact validation by default."""
        requirements = python_tool_requirements(output_path="/tmp/plot.png")
        artifact_reqs = [
            r
            for r in requirements
            if "output file" in requirement_description(r).lower()
        ]
        assert len(artifact_reqs) == 1

    def test_without_output_path_disables_artifact_requirement(self):
        """Artifact validation should be absent without output_path."""
        requirements = python_tool_requirements()
        artifact_reqs = [
            r
            for r in requirements
            if "output file" in requirement_description(r).lower()
        ]
        assert len(artifact_reqs) == 0

    def test_with_allowed_imports_adds_import_requirement(self):
        """Allowed imports should add an import validation requirement."""
        requirements = python_tool_requirements(allowed_imports=["numpy", "matplotlib"])
        import_reqs = [
            r for r in requirements if "import" in requirement_description(r).lower()
        ]
        assert len(import_reqs) > 0


# endregion


# region: Individual requirement validation tests


class TestMustInvokePythonTool:
    """Tests for MustInvokePythonTool requirement."""

    def test_tool_not_called(self):
        """Should fail if python tool not called."""
        req = python_tool_requirements()[0]

        ctx = from_model("Here is the code:\n```python\nprint('hello')\n```")
        result = validation_fn(req)(ctx)

        assert result.as_bool() is False
        reason_lower = validation_reason(result).lower()
        assert "no tool calls" in reason_lower or "did not call" in reason_lower

    def test_python_tool_called(self):
        """Should pass if python tool is called."""
        req = python_tool_requirements()[0]

        ctx = ChatContext()
        output = ModelOutputThunk(
            value="I'll execute this code",
            tool_calls={"python": python_tool_call("print('hi')")},
        )
        ctx = ctx.add(output)

        result = validation_fn(req)(ctx)
        assert result.as_bool() is True


class TestPythonToolHasCodeArg:
    """Tests for PythonToolHasCodeArg requirement."""

    def test_missing_code_argument(self):
        """Should fail if python tool call has no code argument."""
        req = python_tool_requirements()[1]

        ctx = ChatContext()
        output = ModelOutputThunk(
            value="I'll execute this",
            tool_calls={"python": python_tool_call(other="value")},
        )
        ctx = ctx.add(output)

        result = validation_fn(req)(ctx)
        assert result.as_bool() is False
        assert "code" in validation_reason(result).lower()

    def test_has_code_argument(self):
        """Should pass if python tool call has code argument."""
        req = python_tool_requirements()[1]

        ctx = ChatContext()
        output = ModelOutputThunk(
            value="I'll execute this",
            tool_calls={"python": python_tool_call("print('hi')")},
        )
        ctx = ctx.add(output)

        result = validation_fn(req)(ctx)
        assert result.as_bool() is True


class TestCodeParsesRequirement:
    """Tests for code parsing requirement."""

    def test_valid_code(self):
        """Valid code should pass."""
        parse_reqs = requirements_matching("parse")
        parse_req = parse_reqs[0]

        ctx = from_model("```python\nx = 1\nprint(x)\n```")
        result = validation_fn(parse_req)(ctx)

        assert result.as_bool() is True

    def test_syntax_error(self):
        """Syntax errors should be caught."""
        parse_reqs = requirements_matching("parse")
        parse_req = parse_reqs[0]

        ctx = from_model("```python\ndef foo(\n    return 42\n```")
        result = validation_fn(parse_req)(ctx)

        assert result.as_bool() is False
        assert "syntax" in validation_reason(result).lower()

    def test_valid_code_from_tool_calls(self):
        """Valid code in tool_calls should parse."""
        parse_reqs = requirements_matching("parse")
        parse_req = parse_reqs[0]

        # Create context with tool_calls instead of markdown
        ctx = ChatContext()
        output = ModelOutputThunk(
            value="", tool_calls={"python": python_tool_call("x = 1\nprint(x)")}
        )
        ctx = ctx.add(output)

        result = validation_fn(parse_req)(ctx)

        assert result.as_bool() is True

    def test_syntax_error_from_tool_calls(self):
        """Syntax errors in tool_calls should be caught."""
        parse_reqs = requirements_matching("parse")
        parse_req = parse_reqs[0]

        # Create context with tool_calls containing syntax error
        ctx = ChatContext()
        output = ModelOutputThunk(
            value="", tool_calls={"python": python_tool_call("def foo(\n    return 42")}
        )
        ctx = ctx.add(output)

        result = validation_fn(parse_req)(ctx)

        assert result.as_bool() is False
        assert "syntax" in validation_reason(result).lower()


class TestImportAllowlistRequirement:
    """Tests for import allowlist requirement."""

    def test_allowed_imports(self):
        """Allowed imports should pass."""
        allowed = ["numpy", "matplotlib"]
        import_reqs = [
            r
            for r in python_tool_requirements(allowed_imports=allowed)
            if "import" in requirement_description(r).lower()
        ]
        assert len(import_reqs) > 0
        import_req = import_reqs[0]

        ctx = from_model(
            "```python\nimport numpy\nimport matplotlib.pyplot as plt\n```"
        )
        result = validation_fn(import_req)(ctx)

        assert result.as_bool() is True

    def test_unauthorized_imports(self):
        """Unauthorized imports should fail."""
        allowed = ["numpy"]
        import_reqs = [
            r
            for r in python_tool_requirements(allowed_imports=allowed)
            if "import" in requirement_description(r).lower()
        ]
        import_req = import_reqs[0]

        ctx = from_model("```python\nimport subprocess\n```")
        result = validation_fn(import_req)(ctx)

        assert result.as_bool() is False
        assert "subprocess" in validation_reason(result)

    def test_allowed_imports_from_tool_calls(self):
        """Allowed imports in tool_calls should pass."""
        allowed = ["numpy", "matplotlib"]
        import_reqs = [
            r
            for r in python_tool_requirements(allowed_imports=allowed)
            if "import" in requirement_description(r).lower()
        ]
        import_req = import_reqs[0]

        # Create context with tool_calls
        ctx = ChatContext()
        code = "import numpy\nimport matplotlib.pyplot as plt\nprint(numpy.pi)"
        output = ModelOutputThunk(
            value="", tool_calls={"python": python_tool_call(code)}
        )
        ctx = ctx.add(output)

        result = validation_fn(import_req)(ctx)

        assert result.as_bool() is True

    def test_unauthorized_imports_from_tool_calls(self):
        """Unauthorized imports in tool_calls should fail."""
        allowed = ["numpy"]
        import_reqs = [
            r
            for r in python_tool_requirements(allowed_imports=allowed)
            if "import" in requirement_description(r).lower()
        ]
        import_req = import_reqs[0]

        # Create context with tool_calls
        ctx = ChatContext()
        output = ModelOutputThunk(
            value="", tool_calls={"python": python_tool_call("import subprocess\n")}
        )
        ctx = ctx.add(output)

        result = validation_fn(import_req)(ctx)

        assert result.as_bool() is False
        assert "subprocess" in validation_reason(result)


class TestMatplotlibHeadlessRequirement:
    """Tests for matplotlib headless backend requirement."""

    def test_plt_show_without_backend(self):
        """plt.show() without headless backend should fail."""
        matplotlib_reqs = requirements_matching("headless")
        assert len(matplotlib_reqs) > 0
        matplotlib_req = matplotlib_reqs[0]

        ctx = from_model("```python\nplt.plot([1, 2, 3])\nplt.show()\n```")
        result = validation_fn(matplotlib_req)(ctx)

        assert result.as_bool() is False
        assert "headless" in validation_reason(
            result
        ).lower() or "Agg" in validation_reason(result)

    def test_plt_show_with_backend(self):
        """plt.show() with headless backend should pass."""
        matplotlib_reqs = requirements_matching("headless")
        matplotlib_req = matplotlib_reqs[0]

        ctx = from_model(
            "```python\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "plt.plot([1, 2, 3])\n"
            "plt.show()\n"
            "```"
        )
        result = validation_fn(matplotlib_req)(ctx)

        assert result.as_bool() is True

    def test_no_plt_show(self):
        """Code without plt.show() should pass."""
        matplotlib_reqs = requirements_matching("headless")
        matplotlib_req = matplotlib_reqs[0]

        ctx = from_model("```python\nplt.plot([1, 2, 3])\nplt.savefig('plot.png')\n```")
        result = validation_fn(matplotlib_req)(ctx)

        assert result.as_bool() is True


class TestPlotsAreSavedRequirement:
    """Tests for plots must be saved requirement."""

    def test_plot_without_savefig(self):
        """Plotting without savefig should fail."""
        plot_reqs = requirements_matching("savefig")
        assert len(plot_reqs) > 0
        plot_req = plot_reqs[0]

        ctx = from_model("```python\nplt.plot([1, 2, 3])\nplt.show()\n```")
        result = validation_fn(plot_req)(ctx)

        assert result.as_bool() is False
        assert "savefig" in validation_reason(result).lower()

    def test_plot_with_savefig(self):
        """Plotting with savefig should pass."""
        plot_reqs = requirements_matching("savefig")
        plot_req = plot_reqs[0]

        ctx = from_model("```python\nplt.plot([1, 2, 3])\nplt.savefig('plot.png')\n```")
        result = validation_fn(plot_req)(ctx)

        assert result.as_bool() is True

    def test_no_plotting(self):
        """Code without plotting should pass."""
        plot_reqs = requirements_matching("savefig")
        plot_req = plot_reqs[0]

        ctx = from_model("```python\nx = 1\nprint(x)\n```")
        result = validation_fn(plot_req)(ctx)

        assert result.as_bool() is True


class TestOutputArtifactsRequirement:
    """Tests for output artifacts requirement."""

    def test_output_file_not_created(self):
        """Should fail if output file not created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "plot.png")

            artifact_reqs = requirements_matching(
                "output file", output_path=output_path
            )
            assert len(artifact_reqs) > 0
            artifact_req = artifact_reqs[0]

            ctx = from_model("Code ran successfully")
            result = validation_fn(artifact_req)(ctx)

            assert result.as_bool() is False
            assert output_path in validation_reason(result)

    def test_output_file_exists_and_nonempty(self):
        """Should pass if output file exists and is non-empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "plot.png")
            Path(output_path).write_bytes(b"fake png data")

            artifact_reqs = requirements_matching(
                "output file", output_path=output_path
            )
            artifact_req = artifact_reqs[0]

            ctx = from_model("Code ran successfully")
            result = validation_fn(artifact_req)(ctx)

            assert result.as_bool() is True

    def test_output_file_empty(self):
        """Should fail if output file exists but is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "plot.png")
            Path(output_path).write_bytes(b"")

            artifact_reqs = requirements_matching(
                "output file", output_path=output_path
            )
            artifact_req = artifact_reqs[0]

            ctx = from_model("Code ran successfully")
            result = validation_fn(artifact_req)(ctx)

            assert result.as_bool() is False
            assert "empty" in validation_reason(result).lower()

    def test_output_artifact_disabled_without_output_path(self):
        """Output artifact requirement should not be present without output_path."""
        artifact_reqs = requirements_matching("output file")
        assert len(artifact_reqs) == 0


class TestOutputLimitValidator:
    """Tests for _make_output_limit_validator."""

    def test_empty_output_passes(self):
        """No stdout/stderr should pass."""
        ctx = ChatContext().add(ModelOutputThunk(value="response"))
        limit_reqs = requirements_matching("exceed", output_limit_bytes=1000)
        assert len(limit_reqs) > 0
        limit_req = limit_reqs[0]

        result = validation_fn(limit_req)(ctx)
        assert result.as_bool() is True

    def test_output_within_limit_passes(self):
        """Output under limit should pass."""
        output = ModelOutputThunk(value="response")
        setattr(output, "stdout", "x" * 500)
        setattr(output, "stderr", "")
        ctx = ChatContext().add(output)

        limit_reqs = requirements_matching("exceed", output_limit_bytes=1000)
        limit_req = limit_reqs[0]

        result = validation_fn(limit_req)(ctx)
        assert result.as_bool() is True

    def test_stdout_exceeds_limit_fails(self):
        """Stdout exceeding limit should fail."""
        output = ModelOutputThunk(value="response")
        setattr(output, "stdout", "x" * 1500)
        setattr(output, "stderr", "")
        ctx = ChatContext().add(output)

        limit_reqs = requirements_matching("exceed", output_limit_bytes=1000)
        limit_req = limit_reqs[0]

        result = validation_fn(limit_req)(ctx)
        assert result.as_bool() is False
        assert "exceeding" in validation_reason(result).lower()
        assert "1500" in validation_reason(result)

    def test_stderr_exceeds_limit_fails(self):
        """Stderr exceeding limit should fail."""
        output = ModelOutputThunk(value="response")
        setattr(output, "stdout", "")
        setattr(output, "stderr", "e" * 1500)
        ctx = ChatContext().add(output)

        limit_reqs = requirements_matching("exceed", output_limit_bytes=1000)
        limit_req = limit_reqs[0]

        result = validation_fn(limit_req)(ctx)
        assert result.as_bool() is False
        assert "exceeding" in validation_reason(result).lower()

    def test_combined_output_exceeds_limit_fails(self):
        """Combined stdout+stderr exceeding limit should fail."""
        output = ModelOutputThunk(value="response")
        setattr(output, "stdout", "x" * 600)
        setattr(output, "stderr", "e" * 600)  # Combined: 1200 bytes
        ctx = ChatContext().add(output)

        limit_reqs = requirements_matching("exceed", output_limit_bytes=1000)
        limit_req = limit_reqs[0]

        result = validation_fn(limit_req)(ctx)
        assert result.as_bool() is False
        assert "1200" in validation_reason(result)

    def test_utf8_multibyte_characters_counted_correctly(self):
        """Multibyte UTF-8 characters should be counted in bytes, not chars."""
        output = ModelOutputThunk(value="response")
        setattr(output, "stdout", "🎉" * 100)  # 4 bytes per emoji = 400 bytes
        setattr(output, "stderr", "")
        ctx = ChatContext().add(output)

        limit_reqs = requirements_matching("exceed", output_limit_bytes=300)
        limit_req = limit_reqs[0]

        result = validation_fn(limit_req)(ctx)
        assert result.as_bool() is False  # 400 > 300
        assert "exceeding" in validation_reason(result).lower()

    def test_limit_at_boundary(self):
        """Output exactly at limit should pass."""
        output = ModelOutputThunk(value="response")
        setattr(output, "stdout", "x" * 1000)  # Exactly 1000 bytes
        setattr(output, "stderr", "")
        ctx = ChatContext().add(output)

        limit_reqs = requirements_matching("exceed", output_limit_bytes=1000)
        limit_req = limit_reqs[0]

        result = validation_fn(limit_req)(ctx)
        assert result.as_bool() is True

    def test_limit_one_byte_over_boundary_fails(self):
        """Output one byte over limit should fail."""
        output = ModelOutputThunk(value="response")
        setattr(output, "stdout", "x" * 1001)  # 1001 bytes
        setattr(output, "stderr", "")
        ctx = ChatContext().add(output)

        limit_reqs = requirements_matching("exceed", output_limit_bytes=1000)
        limit_req = limit_reqs[0]

        result = validation_fn(limit_req)(ctx)
        assert result.as_bool() is False


# endregion
