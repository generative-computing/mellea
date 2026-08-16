# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that building the `m` CLI does not import heavy dependencies.

Every `m` invocation imports `cli.m` to construct the Typer command tree, so
anything imported at module scope there is paid on `m --help` and on shell tab
completion. Sub-command implementations must therefore defer their imports into
the command body (see `cli.serve.commands.serve` for the pattern).

These run in a subprocess because the test session itself imports `mellea`,
which would poison an in-process `sys.modules` check.
"""

import subprocess
import sys
import textwrap

# Packages that must not be pulled in merely by constructing the CLI.
_FORBIDDEN_MODULES = [
    "mellea",
    "litellm",
    "nltk",
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "fastapi",
    "uvicorn",
]


def _import_cli_in_subprocess(target: str) -> set[str]:
    """Import *target* in a fresh interpreter, returning forbidden loaded modules.

    Args:
        target: Dotted module path to import.

    Returns:
        The subset of `_FORBIDDEN_MODULES` present in `sys.modules` afterwards.
    """
    script = textwrap.dedent(f"""
        import importlib, sys
        importlib.import_module({target!r})
        forbidden = {_FORBIDDEN_MODULES!r}
        print(",".join(m for m in forbidden if m in sys.modules))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return {name for name in result.stdout.strip().split(",") if name}


def test_cli_entrypoint_does_not_import_heavy_deps():
    """Importing `cli.m` must not load `mellea` or other heavy dependencies."""
    loaded = _import_cli_in_subprocess("cli.m")
    assert loaded == set(), (
        f"cli.m eagerly imports {sorted(loaded)}. Move the import into the "
        "command body so `m --help` stays fast."
    )


def test_help_output_does_not_import_heavy_deps():
    """`m --help` must run without loading heavy dependencies.

    Guards the whole path Typer walks to render help — including resolving
    default values and enums for every sub-command's options — not just the
    module-level imports covered above.
    """
    script = textwrap.dedent(f"""
        import sys
        from typer.testing import CliRunner
        from cli.m import cli

        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0, result.output
        forbidden = {_FORBIDDEN_MODULES!r}
        print(",".join(m for m in forbidden if m in sys.modules))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    loaded = {name for name in result.stdout.strip().split(",") if name}
    assert loaded == set(), f"`m --help` eagerly imports {sorted(loaded)}."
