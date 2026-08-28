# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that the package-level `__version__` is exposed and stays in sync with pyproject.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

import mellea

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_version_attribute_exists():
    """`mellea.__version__` is a non-empty string."""
    assert isinstance(mellea.__version__, str)
    assert mellea.__version__


def test_version_matches_pyproject():
    """The runtime `__version__` matches the `version` declared in pyproject.toml."""
    with PYPROJECT.open("rb") as f:
        pyproject = tomllib.load(f)
    assert mellea.__version__ == pyproject["project"]["version"]
