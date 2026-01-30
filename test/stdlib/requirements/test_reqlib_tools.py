import pytest

from mellea.stdlib.requirements.tool_reqs import _name2str


def test_name2str():
    """Test handling when no Python code is present."""

    def test123():
        pass

    assert _name2str(test123) == "test123"
    assert _name2str("test1234") == "test1234"
