"""Allows you to use `pytest docs` to run the examples.

To run notebooks, use: uv run --with 'mcp' pytest --nbmake docs/examples/notebooks/
"""

import ast
import os
import pathlib
import subprocess
import sys

import pytest

# Import system capability detection from test/conftest.py
# Add test directory to path to enable import
_test_dir = pathlib.Path(__file__).parent.parent.parent / "test"
_test_dir_abs = _test_dir.resolve()
if str(_test_dir_abs) not in sys.path:
    sys.path.insert(0, str(_test_dir_abs))

try:
    # Import with explicit module name to avoid conflicts
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_conftest", _test_dir_abs / "conftest.py"
    )
    if spec and spec.loader:
        test_conftest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_conftest)
        get_system_capabilities = test_conftest.get_system_capabilities
    else:
        raise ImportError("Could not load test/conftest.py")
except (ImportError, AttributeError) as e:
    # Fallback if test/conftest.py not available
    import warnings

    warnings.warn(
        f"Could not import get_system_capabilities from test/conftest.py: {e}. Heavy RAM tests will NOT be skipped!"
    )

    def get_system_capabilities():
        return {
            "has_gpu": False,
            "gpu_memory_gb": 0,
            "ram_gb": 0,
            "has_api_keys": {},
            "has_ollama": False,
        }


examples_to_skip = {
    "__init__.py",
    "simple_rag_with_filter.py",
    "mcp_example.py",
    "client.py",
    "pii_serve.py",
}


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Append the skipped examples if needed.
    if len(examples_to_skip) == 0:
        return

    terminalreporter.ensure_newline()
    terminalreporter.section("Skipped Examples", sep="=", blue=True, bold=True)
    newline = "\n"
    terminalreporter.line(
        f"Examples with the following names were skipped because they cannot be easily run in the pytest framework; please run them manually:\n{newline.join(examples_to_skip)}"
    )


# This doesn't replace the existing pytest file collection behavior.
def pytest_collect_file(parent: pytest.Dir, file_path: pathlib.PosixPath):
    # Do a quick check that it's a .py file in the expected `docs/examples` folder. We can make
    # this more exact if needed.
    if (
        file_path.suffix == ".py"
        and "docs" in file_path.parts
        and "examples" in file_path.parts
    ):
        # Skip this test. It requires additional setup.
        if file_path.name in examples_to_skip:
            return

        return ExampleFile.from_parent(parent, path=file_path)


class ExampleFile(pytest.File):
    def collect(self):
        return [ExampleItem.from_parent(self, name=self.name)]


class ExampleItem(pytest.Item):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def runtest(self):
        process = subprocess.Popen(
            [sys.executable, self.path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Enable line-buffering
        )

        # Capture stdout output and output it so it behaves like a regular test with -s.
        stdout_lines = []
        if process.stdout is not None:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()  # Ensure the output is printed immediately
                stdout_lines.append(line)
            process.stdout.close()

        retcode = process.wait()

        # Capture stderr output.
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read()

        if retcode != 0:
            # Check if this is a pytest.skip() call (indicated by "Skipped:" in stderr)
            if "Skipped:" in stderr or "_pytest.outcomes.Skipped" in stderr:
                # Extract skip reason from stderr
                skip_reason = "Example skipped"
                for line in stderr.split("\n"):
                    if line.startswith("Skipped:"):
                        skip_reason = line.replace("Skipped:", "").strip()
                        break
                pytest.skip(skip_reason)
            else:
                raise ExampleTestException(
                    f"Example failed with exit code {retcode}.\nStderr: {stderr}\n"
                )

    def repr_failure(self, excinfo, style=None):
        """Called when self.runtest() raises an exception."""
        if isinstance(excinfo.value, ExampleTestException):
            return str(excinfo.value)

        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, f"usecase: {self.name}"


class ExampleTestException(Exception):
    """Custom exception for error reporting."""


def pytest_runtest_setup(item):
    """Apply skip logic to ExampleItem objects based on system capabilities.

    This ensures examples respect the same capability checks as regular tests
    (RAM, GPU, Ollama, API keys, etc.).
    """
    if not isinstance(item, ExampleItem):
        return

    # Get system capabilities
    capabilities = get_system_capabilities()

    # Get gh_run status (CI environment)
    gh_run = int(os.environ.get("CICD", 0))

    # Get config options (all default to False for examples)
    ignore_all = False
    ignore_gpu = False
    ignore_ram = False
    ignore_ollama = False
    ignore_api_key = False

    # Skip qualitative tests in CI
    if item.get_closest_marker("qualitative") and gh_run == 1:
        pytest.skip(
            reason="Skipping qualitative test: got env variable CICD == 1. Used only in gh workflows."
        )

    # Skip tests requiring API keys if not available
    if item.get_closest_marker("requires_api_key") and not ignore_api_key:
        for backend in ["openai", "watsonx"]:
            if item.get_closest_marker(backend):
                if not capabilities["has_api_keys"].get(backend):
                    pytest.skip(
                        f"Skipping test: {backend} API key not found in environment"
                    )

    # Skip tests requiring GPU if not available
    if item.get_closest_marker("requires_gpu") and not ignore_gpu:
        if not capabilities["has_gpu"]:
            pytest.skip("Skipping test: GPU not available")

    # Skip tests requiring heavy RAM if insufficient
    if item.get_closest_marker("requires_heavy_ram") and not ignore_ram:
        RAM_THRESHOLD_GB = 48  # Based on real-world testing
        if capabilities["ram_gb"] > 0 and capabilities["ram_gb"] < RAM_THRESHOLD_GB:
            pytest.skip(
                f"Skipping test: Insufficient RAM ({capabilities['ram_gb']:.1f}GB < {RAM_THRESHOLD_GB}GB)"
            )

    # Backend-specific skipping
    if item.get_closest_marker("watsonx") and not ignore_api_key:
        if not capabilities["has_api_keys"].get("watsonx"):
            pytest.skip(
                "Skipping test: Watsonx API credentials not found in environment"
            )

    if item.get_closest_marker("vllm") and not ignore_gpu:
        if not capabilities["has_gpu"]:
            pytest.skip("Skipping test: vLLM requires GPU")

    if item.get_closest_marker("ollama") and not ignore_ollama:
        if not capabilities["has_ollama"]:
            pytest.skip(
                "Skipping test: Ollama not available (port 11434 not listening)"
            )


def pytest_collection_modifyitems(items):
    """Apply markers from example files to ExampleItem objects.

    The custom ExampleFile/ExampleItem collection doesn't automatically
    inherit pytestmark from the Python modules, so we need to parse
    the files and apply markers manually.
    """
    for item in items:
        if isinstance(item, ExampleItem):
            # Read the file and look for pytestmark
            try:
                with open(item.path) as f:
                    tree = ast.parse(f.read(), filename=str(item.path))

                # Look for pytestmark assignment
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "pytestmark"
                            ):
                                # Extract marker names from the assignment
                                markers = _extract_markers_from_node(node.value)
                                for marker_name in markers:
                                    item.add_marker(getattr(pytest.mark, marker_name))
            except Exception:
                # If we can't parse the file, skip marker application
                pass


def _extract_markers_from_node(node):
    """Extract marker names from a pytestmark assignment node."""
    markers = []

    if isinstance(node, ast.Attribute):
        # Single marker: pytest.mark.ollama
        if isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
            markers.append(node.attr)
    elif isinstance(node, ast.List):
        # List of markers: [pytest.mark.ollama, pytest.mark.llm]
        for elt in node.elts:
            if isinstance(elt, ast.Attribute):
                if isinstance(elt.value, ast.Attribute) and elt.value.attr == "mark":
                    markers.append(elt.attr)

    return markers
