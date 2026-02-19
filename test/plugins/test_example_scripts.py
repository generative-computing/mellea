"""Tests that all docs/examples/plugins example scripts run without throwing errors.

Each test runs the example script end-to-end using runpy.run_path (with
run_name="__main__" so the if __name__ == "__main__": block executes) using a
mock backend so no real LLM API calls are made.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mcpgateway.plugins.framework")

from mellea.core.backend import Backend
from mellea.core.base import GenerateLog, ModelOutputThunk
from mellea.plugins.manager import shutdown_plugins
from mellea.stdlib.context import SimpleContext


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "docs" / "examples" / "plugins"


class _MockBackend(Backend):
    """Minimal backend that returns a faked ModelOutputThunk — no LLM API calls."""

    model_id = "mock-model"

    def __init__(self, *args, **kwargs):
        pass

    async def generate_from_context(self, action, ctx, **kwargs):
        mot = MagicMock(spec=ModelOutputThunk)
        glog = GenerateLog()
        glog.prompt = "mocked"
        mot._generate_log = glog
        mot.parsed_repr = None
        mot.tool_calls = None  # required by uses_tool() requirement validation

        async def _avalue():
            return "mocked output"

        mot.avalue = _avalue
        mot.value = "mocked output string"
        return mot, SimpleContext()

    async def generate_from_raw(self, actions, ctx, **kwargs):
        return []


@pytest.fixture(autouse=True)
async def reset_plugins():
    """Reset the global plugin registry after every test."""
    yield
    await shutdown_plugins()


def test_standalone_hooks_example():
    """standalone_hooks.py runs without error using a mock backend."""
    with patch(
        "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
    ):
        runpy.run_path(str(EXAMPLES_DIR / "standalone_hooks.py"), run_name="__main__")


def test_class_plugin_example():
    """class_plugin.py runs without error using a mock backend."""
    with patch(
        "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
    ):
        runpy.run_path(str(EXAMPLES_DIR / "class_plugin.py"), run_name="__main__")


def test_plugin_set_composition_example():
    """plugin_set_composition.py runs without error using a mock backend."""
    with patch(
        "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
    ):
        runpy.run_path(
            str(EXAMPLES_DIR / "plugin_set_composition.py"), run_name="__main__"
        )


def test_plugin_scoped_example():
    """plugin_scoped.py runs without error using a mock backend."""
    with patch(
        "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
    ):
        runpy.run_path(str(EXAMPLES_DIR / "plugin_scoped.py"), run_name="__main__")


def test_session_scoped_example():
    """session_scoped.py runs without error using a mock backend."""
    with patch(
        "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
    ):
        runpy.run_path(str(EXAMPLES_DIR / "session_scoped.py"), run_name="__main__")


def test_tool_hooks_example():
    """tool_hooks.py runs without error using a mock backend and mocked _call_tools.

    _call_tools is mocked to return a single fake tool output so that the
    example's "expected tool call" assertions in Scenarios 1 and 3 do not call
    sys.exit(1).  Scenarios 2 and 4 (where the example expects an empty list)
    will log a warning instead, which is acceptable for this smoke test.
    """
    mock_output = MagicMock()
    mock_output.content = "mocked tool output"

    with (
        patch("mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend),
        patch("mellea.stdlib.functional._call_tools", return_value=[mock_output]),
    ):
        runpy.run_path(str(EXAMPLES_DIR / "tool_hooks.py"), run_name="__main__")
