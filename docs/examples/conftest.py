"""Allows you to use `pytest docs` to run the examples.

To run notebooks, use: uv run --with 'mcp' pytest --nbmake docs/examples/notebooks/
"""

import ast
import os
import pathlib
import subprocess
import sys

import pytest

# Lazy import of system capability detection to avoid circular imports
_get_system_capabilities = None


def get_system_capabilities():
    """Lazy load system capabilities from test/conftest.py."""
    global _get_system_capabilities

    if _get_system_capabilities is not None:
        return _get_system_capabilities()

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
            _get_system_capabilities = test_conftest.get_system_capabilities
            return _get_system_capabilities()
        else:
            raise ImportError("Could not load test/conftest.py")
    except (ImportError, AttributeError) as e:
        # Fallback if test/conftest.py not available
        import warnings

        warnings.warn(
            f"Could not import get_system_capabilities from test/conftest.py: {e}. Heavy RAM tests will NOT be skipped!"
        )

        def fallback():
            return {
                "has_gpu": False,
                "gpu_memory_gb": 0,
                "ram_gb": 0,
                "has_api_keys": {},
                "has_ollama": False,
            }

        _get_system_capabilities = fallback
        return fallback()


examples_to_skip: dict[str, str] = {}


def _extract_markers_from_file(file_path):
    """Extract pytest markers from comment in file without parsing Python.

    Looks for lines like: # pytest: marker1, marker2, marker3
    Returns list of marker names.
    """
    try:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("# pytest:"):
                    marker_text = line[9:].strip()  # Remove "# pytest:"
                    return [m.strip() for m in marker_text.split(",") if m.strip()]
                # Stop after first few lines (markers should be at top)
                if (
                    len(line) > 0
                    and not line.startswith("#")
                    and not line.startswith('"""')
                ):
                    break
    except Exception:
        pass
    return []


def _should_skip_collection(markers):
    """Check if example should be skipped during collection based on markers.

    Returns (should_skip, reason) tuple.
    """
    if not markers:
        return False, None

    try:
        capabilities = get_system_capabilities()
    except Exception:
        # If we can't get capabilities, don't skip (fail open)
        return False, None

    gh_run = int(os.environ.get("CICD", 0))

    # Skip tests marked with skip_always
    if "skip_always" in markers:
        return True, "Example marked to always skip (skip_always marker)"

    # Skip qualitative tests in CI
    if "qualitative" in markers and gh_run == 1:
        return True, "Skipping qualitative test in CI (CICD=1)"

    # Skip slow tests if SKIP_SLOW=1 environment variable is set
    if "slow" in markers and int(os.environ.get("SKIP_SLOW", 0)) == 1:
        return True, "Skipping slow test (SKIP_SLOW=1)"

    # Skip tests requiring heavy RAM if insufficient
    if "requires_heavy_ram" in markers:
        RAM_THRESHOLD_GB = 48
        if capabilities["ram_gb"] > 0 and capabilities["ram_gb"] < RAM_THRESHOLD_GB:
            return (
                True,
                f"Insufficient RAM ({capabilities['ram_gb']:.1f}GB < {RAM_THRESHOLD_GB}GB)",
            )

    # Skip tests requiring GPU if not available
    if "requires_gpu" in markers or "vllm" in markers:
        if not capabilities["has_gpu"]:
            return True, "GPU not available"

    # Skip tests requiring Ollama if not available
    if "ollama" in markers:
        if not capabilities["has_ollama"]:
            return True, "Ollama not available (port 11434 not listening)"

    # Skip tests requiring API keys
    if "requires_api_key" in markers or "watsonx" in markers:
        if "watsonx" in markers and not capabilities["has_api_keys"].get("watsonx"):
            return True, "Watsonx API credentials not found"
        if "openai" in markers and not capabilities["has_api_keys"].get("openai"):
            return True, "OpenAI API key not found"

    return False, None


def _check_optional_imports(file_path):
    """Check if file has optional imports that aren't installed.

    Returns (should_skip, reason) tuple.
    """
    try:
        with open(file_path) as f:
            content = f.read()

        # Check for langchain imports
        if "from langchain" in content or "import langchain" in content:
            try:
                import langchain_core
            except ImportError:
                return True, "langchain_core not installed"

    except Exception:
        pass

    return False, None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Append the skipped examples if needed.
    if len(examples_to_skip) == 0:
        return

    terminalreporter.ensure_newline()
    terminalreporter.section("Skipped Examples", sep="=", blue=True, bold=True)
    terminalreporter.line("The following examples were skipped during collection:\n")
    for filename, reason in examples_to_skip.items():
        terminalreporter.line(f"  • {filename}: {reason}")


def pytest_ignore_collect(collection_path, path, config):
    """Ignore files before pytest even tries to parse them.

    This is called BEFORE pytest_collect_file, so we can prevent
    heavy files from being parsed at all.
    """
    # Skip conftest.py itself - it's not a test
    if collection_path.name == "conftest.py":
        return True

    # Only check Python files in docs/examples
    if (
        collection_path.suffix == ".py"
        and "docs" in collection_path.parts
        and "examples" in collection_path.parts
    ):
        # Skip files in the manual skip list
        if collection_path.name in examples_to_skip:
            return True

        # Extract markers and check if we should skip
        try:
            markers = _extract_markers_from_file(collection_path)
            should_skip, reason = _should_skip_collection(markers)
            if should_skip and reason:
                # Add to skip list with reason for terminal summary
                examples_to_skip[collection_path.name] = reason
                # Return True to ignore this file completely
                return True
        except Exception:
            # If anything goes wrong, don't skip
            pass

    return False


def pytest_pycollect_makemodule(module_path, path, parent):
    """Prevent pytest from importing Python modules as test modules.

    This hook is called BEFORE pytest tries to import the module,
    so we can prevent import errors from optional dependencies.
    """
    # Only handle files in docs/examples
    if (
        module_path.suffix == ".py"
        and "docs" in module_path.parts
        and "examples" in module_path.parts
    ):
        # Check for optional imports
        should_skip, reason = _check_optional_imports(module_path)
        if should_skip and reason:
            # Add to skip list with reason for terminal summary and return None to prevent module creation
            examples_to_skip[module_path.name] = reason
            return None

    # Return None to let pytest handle it normally
    return None


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

        # Check for optional imports before creating ExampleFile
        should_skip, _reason = _check_optional_imports(file_path)
        if should_skip:
            return None

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

    Parses comment-based markers from example files in the format:
        # pytest: marker1, marker2, marker3

    This keeps examples clean while allowing intelligent test skipping.
    """
    for item in items:
        if isinstance(item, ExampleItem):
            # Read the file and look for comment-based markers
            try:
                with open(item.path) as f:
                    for line in f:
                        line = line.strip()
                        # Look for comment-based marker line
                        if line.startswith("# pytest:"):
                            # Extract markers after "# pytest:"
                            marker_text = line[9:].strip()  # Remove "# pytest:"
                            markers = [m.strip() for m in marker_text.split(",")]
                            for marker_name in markers:
                                if marker_name:  # Skip empty strings
                                    item.add_marker(getattr(pytest.mark, marker_name))
                            break  # Only process first pytest comment line
            except Exception:
                # If we can't parse the file, skip marker application
                pass
