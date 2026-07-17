# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `.github/scripts/bump_version.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "bump_version.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bump_version", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bump_version"] = mod
    spec.loader.exec_module(mod)
    return mod


bump_version = _load_module()


@pytest.mark.parametrize(
    ("current", "mode", "expected"),
    [
        ("0.6.0rc0", "rc", "0.6.0rc1"),
        ("0.6.0rc5", "rc", "0.6.0rc6"),
        ("0.6.0rc0", "final", "0.6.0"),
        ("0.6.0rc2", "final", "0.6.0"),
        ("0.6.0", "patch-rc", "0.6.1rc0"),
        ("0.6.1rc0", "patch-rc", "0.6.1rc1"),
        ("0.6.1rc3", "patch-rc", "0.6.1rc4"),
        ("0.6.1rc0", "patch-final", "0.6.1"),
        ("0.6.5rc2", "patch-final", "0.6.5"),
        ("0.6.0.dev0", "dev", "0.6.0.dev1"),
        ("0.7.0.dev3", "dev", "0.7.0.dev4"),
    ],
)
def test_compute_next_happy_paths(current, mode, expected):
    got = bump_version.compute_next(Version(current), mode)
    assert str(got) == expected


@pytest.mark.parametrize(
    ("current", "mode"),
    [
        # rc requires current rc
        ("0.6.0", "rc"),
        # final requires current rc, not a final
        ("0.6.0", "final"),
        # final is for minor rcs (patch=0), not patch rcs
        ("0.6.1rc0", "final"),
        # patch-rc refuses minor rcs (patch=0)
        ("0.6.0rc0", "patch-rc"),
        # patch-final refuses minor rcs
        ("0.6.0rc0", "patch-final"),
        # patch-final requires current rc
        ("0.6.1", "patch-final"),
        # dev versions should never appear on release branches
        ("0.6.0.dev0", "rc"),
        ("0.6.0.dev0", "final"),
        # dev mode requires a .dev release; refuses finals and rcs
        ("0.6.0", "dev"),
        ("0.6.0rc0", "dev"),
    ],
)
def test_compute_next_rejects_disallowed(current, mode):
    with pytest.raises(ValueError):
        bump_version.compute_next(Version(current), mode)


def test_compute_next_unknown_mode():
    with pytest.raises(ValueError):
        bump_version.compute_next(Version("0.6.0rc0"), "bogus")


def test_write_pyproject_roundtrip(tmp_path, monkeypatch):
    fake = tmp_path / "pyproject.toml"
    fake.write_text('[project]\nname = "x"\nversion = "0.6.0rc0"\ndependencies = []\n')
    monkeypatch.setattr(bump_version, "PYPROJECT", fake)
    bump_version.write_pyproject(Version("0.6.0rc1"))
    assert 'version = "0.6.0rc1"' in fake.read_text()
    # only the version line changed, everything else is preserved
    assert 'name = "x"' in fake.read_text()
