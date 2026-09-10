# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that the core `pillow` dependency specifier excludes the broken 12.3.0 release.

Pillow 12.3.0 removed `_Ink` from `PIL._typing`, which breaks `from PIL import
ImageText` and cascades through `docling_core` into `RichDocument` (issue #1640).
These tests assert the specifier in `pyproject.toml` cannot resolve 12.3.0 while
still admitting the known-good releases.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _pillow_specifier():
    """Return the `SpecifierSet` for the core `pillow` dependency in pyproject.toml."""
    with PYPROJECT.open("rb") as f:
        pyproject = tomllib.load(f)
    for dep in pyproject["project"]["dependencies"]:
        req = Requirement(dep)
        if req.name == "pillow":
            return req.specifier
    raise AssertionError("pillow not found in core dependencies")


def test_pillow_specifier_excludes_broken_1230():
    """The pillow specifier must not admit the broken 12.3.0 release (issue #1640)."""
    assert "12.3.0" not in _pillow_specifier()


def test_pillow_specifier_allows_working_versions():
    """The pillow specifier still admits the known-good 12.2.0 release."""
    spec = _pillow_specifier()
    assert "12.2.0" in spec
