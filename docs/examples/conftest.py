"""Allows you to use `pytest docs` to run the examples."""

import pathlib
import subprocess
import sys

import pytest

# Enable nbmake for notebook testing when running pytest in this directory.
# This allows `pytest docs/` to automatically run notebooks via nbmake.
pytest_plugins = ["nbmake"]


def pytest_configure(config):
    """Configure nbmake to run notebooks in docs/examples/notebooks/."""
    # Only enable nbmake if we're collecting from docs directory
    if hasattr(config.option, "nbmake"):
        config.option.nbmake = True


examples_to_skip = {
    "101_example.py",
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
    terminalreporter.line(
        f"Examples with the following names were skipped because they cannot be easily run in the pytest framework; please run them manually:\n{'\n'.join(examples_to_skip)}"
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
            raise ExampleTestException(
                (f"Example failed with exit code {retcode}.\nStderr: {stderr}\n")
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
