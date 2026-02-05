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


examples_to_skip = {
    "__init__.py",
    "simple_rag_with_filter.py",
    "mcp_example.py",
    "client.py",
    "pii_serve.py",
    "mellea_pdf.py",  # External URL returns 403 Forbidden
    "m_decomp_result.py",
    "python_decompose_result.py",
}


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


def pytest_addoption(parser):
    """Add command-line options for skipping capability checks.

    These match the options in test/conftest.py to provide consistent behavior.
    Only adds options if they don't already exist (to avoid conflicts when both
    test/ and docs/ conftest files are loaded).
    """

    # Helper to safely add option only if it doesn't exist
    def add_option_safe(option_name, **kwargs):
        try:
            parser.addoption(option_name, **kwargs)
        except ValueError:
            # Option already exists (likely from test/conftest.py)
            pass

    add_option_safe(
        "--ignore-gpu-check",
        action="store_true",
        default=False,
        help="Ignore GPU requirement checks (examples may fail without GPU)",
    )
    add_option_safe(
        "--ignore-ram-check",
        action="store_true",
        default=False,
        help="Ignore RAM requirement checks (examples may fail with insufficient RAM)",
    )
    add_option_safe(
        "--ignore-ollama-check",
        action="store_true",
        default=False,
        help="Ignore Ollama availability checks (examples will fail if Ollama not running)",
    )
    add_option_safe(
        "--ignore-api-key-check",
        action="store_true",
        default=False,
        help="Ignore API key checks (examples will fail without valid API keys)",
    )
    add_option_safe(
        "--ignore-all-checks",
        action="store_true",
        default=False,
        help="Ignore all requirement checks (GPU, RAM, Ollama, API keys)",
    )


def _collect_vllm_example_files(session) -> list[str]:
    """Collect all example files that have vLLM marker.

    Returns list of file paths.
    """
    vllm_files = set()

    for item in session.items:
        # Check if this is an ExampleItem with vllm marker
        if hasattr(item, "path"):
            file_path = str(item.path)
            # Check if file has vllm marker
            if file_path.endswith(".py"):
                markers = _extract_markers_from_file(file_path)
                if "vllm" in markers:
                    vllm_files.add(file_path)

    return sorted(vllm_files)


def _run_vllm_examples_isolated(session, vllm_files: list[str]) -> int:
    """Run vLLM example files in separate processes for GPU memory isolation.

    Returns exit code (0 = all passed, 1 = any failed).
    """
    print("\n" + "=" * 70)
    print("vLLM Process Isolation Active (Examples)")
    print("=" * 70)
    print(f"Running {len(vllm_files)} vLLM example(s) in separate processes")
    print("to ensure GPU memory is fully released between examples.\n")

    # Set environment variables for vLLM
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    all_passed = True

    for i, file_path in enumerate(vllm_files, 1):
        print(f"\n[{i}/{len(vllm_files)}] Running: {file_path}")
        print("-" * 70)

        # Run example directly with Python
        cmd = [sys.executable, file_path]

        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            all_passed = False
            print(f"✗ Example failed: {file_path}")
        else:
            print(f"✓ Example passed: {file_path}")

    print("\n" + "=" * 70)
    if all_passed:
        print("All vLLM examples passed!")
    else:
        print("Some vLLM examples failed.")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


def pytest_collection_finish(session):
    """After collection, check if we need vLLM process isolation for examples.

    If vLLM examples are collected and there are multiple files,
    run them in separate processes and exit.
    """
    # Only check for examples in docs/examples
    if not any(
        "docs" in str(item.path) and "examples" in str(item.path)
        for item in session.items
    ):
        return

    # Collect vLLM example files
    vllm_files = _collect_vllm_example_files(session)

    # Only use process isolation if multiple vLLM examples
    if len(vllm_files) <= 1:
        return

    # Run examples in isolation
    exit_code = _run_vllm_examples_isolated(session, vllm_files)

    # Clear collected items so pytest doesn't run them again
    session.items.clear()

    # Exit with appropriate code
    pytest.exit("vLLM examples completed in isolated processes", returncode=exit_code)


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
            should_skip, _reason = _should_skip_collection(markers)
            if should_skip:
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
        should_skip, _reason = _check_optional_imports(module_path)
        if should_skip:
            # Add to skip list and return None to prevent module creation
            examples_to_skip.add(module_path.name)
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

    # Get config options from CLI (matching test/conftest.py behavior)
    config = item.config
    ignore_all = config.getoption("--ignore-all-checks", default=False)
    ignore_gpu = config.getoption("--ignore-gpu-check", default=False) or ignore_all
    ignore_ram = config.getoption("--ignore-ram-check", default=False) or ignore_all
    ignore_ollama = (
        config.getoption("--ignore-ollama-check", default=False) or ignore_all
    )
    ignore_api_key = (
        config.getoption("--ignore-api-key-check", default=False) or ignore_all
    )

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
