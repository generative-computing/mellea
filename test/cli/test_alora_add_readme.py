# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `m alora add-readme` layout detection wiring.

The README generator's LLM and the Hugging Face upload are replaced so these
tests exercise only the layout resolution in `cli/alora/commands.py`.
"""

import re

import pytest
from typer.testing import CliRunner

pytest.importorskip("huggingface_hub", reason="huggingface_hub not installed")

from cli.m import cli

runner = CliRunner()


class _FakeHfApi:
    """Stand-in for `HfApi` exposing a canned repository file listing."""

    def __init__(self, repo_files: list[str]) -> None:
        self._repo_files = repo_files

    def list_repo_files(self, **_kwargs) -> list[str]:
        return self._repo_files


def _plain(output: str) -> str:
    """Flatten Rich-rendered CLI output: drop ANSI codes and box lines, join wraps."""
    no_ansi = re.sub(r"\x1b\[[0-9;]*m", "", output)
    return " ".join(no_ansi.replace("│", " ").split())


def _invoke_add_readme(
    monkeypatch,
    extra_args: list[str],
    *,
    interactive: bool = True,
    token: str | None = "test-token",
):
    """Run `add-readme` with generation and uploading stubbed out.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        extra_args: Additional CLI arguments appended after `--name`.
        interactive: Whether standard input counts as a terminal.
        token: Hugging Face token returned by `get_token`, or `None` for none.

    Returns:
        Tuple of the CLI result and the kwargs passed to `generate_readme`.
    """
    captured: dict = {}

    def fake_generate_readme(**kwargs):
        captured.update(kwargs)
        with open(kwargs["output_path"], "w") as fh:
            fh.write("# README\n")
        return kwargs["output_path"]

    monkeypatch.setattr("cli.alora.commands._stdin_is_interactive", lambda: interactive)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: token)
    monkeypatch.setattr("huggingface_hub.create_repo", lambda **_kwargs: None)
    monkeypatch.setattr("huggingface_hub.upload_file", lambda **_kwargs: None)
    monkeypatch.setattr(
        "cli.alora.readme_generator.generate_readme", fake_generate_readme
    )
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "yes")

    result = runner.invoke(
        cli,
        [
            "alora",
            "add-readme",
            "data.jsonl",
            "--basemodel",
            "ibm-granite/granite-4.1-3b",
            "--name",
            "acme/stembolts",
            *extra_args,
        ],
    )
    return result, captured


@pytest.mark.unit
@pytest.mark.parametrize("layout", ["intrinsic", "flat"])
def test_explicit_layout_skips_detection(monkeypatch, layout):
    """`--layout` is forwarded to the generator without inspecting the repo."""
    result, captured = _invoke_add_readme(monkeypatch, ["--layout", layout])

    assert result.exit_code == 0, result.output
    assert captured["layout"] == layout


@pytest.mark.unit
@pytest.mark.parametrize(
    ("repo_files", "expected_layout"),
    [
        (["stembolts/granite-4.1-3b/alora/adapter_config.json"], "intrinsic"),
        (["adapter_config.json", "adapter_model.safetensors"], "flat"),
    ],
)
def test_auto_detects_repo_layout(monkeypatch, repo_files, expected_layout):
    """`--layout auto` uses the repository's file listing to pick the sample."""
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(repo_files))

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    assert captured["layout"] == expected_layout


@pytest.mark.unit
def test_unrecognized_repo_layout_asks_user(monkeypatch):
    """An empty/unrecognized repo prompts for the intended upload layout."""
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(["README.md"]))

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    # The stubbed `input` answers "yes" to the --intrinsic upload question.
    assert captured["layout"] == "intrinsic"
    assert "has no recognizable adapter layout" in result.output


@pytest.mark.unit
def test_unknown_layout_without_terminal_fails_before_generation(monkeypatch):
    """With nobody to ask, an undetectable layout fails fast and names `--layout`.

    Waiting on `input()` here would hang or die with `EOFError` only after the
    slow README generation, so the command must stop before generating.
    """
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(["README.md"]))

    result, captured = _invoke_add_readme(monkeypatch, [], interactive=False)

    assert result.exit_code != 0
    message = _plain(result.output)
    assert "no terminal to ask" in message
    assert "--layout intrinsic" in message
    assert "--layout flat" in message
    assert captured == {}


@pytest.mark.unit
def test_missing_token_without_terminal_fails_before_generation(monkeypatch):
    """No token means no repo inspection, which falls through to the same failure."""
    result, captured = _invoke_add_readme(
        monkeypatch, [], interactive=False, token=None
    )

    assert result.exit_code != 0
    assert "no terminal to ask" in _plain(result.output)
    assert captured == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra_args", "repo_files", "expected_layout"),
    [
        (["--layout", "flat"], None, "flat"),
        (["--layout", "intrinsic"], None, "intrinsic"),
        ([], ["stembolts/granite-4.1-3b/alora/adapter_config.json"], "intrinsic"),
        ([], ["adapter_config.json"], "flat"),
    ],
)
def test_no_terminal_needs_no_prompt_when_layout_is_known(
    monkeypatch, extra_args, repo_files, expected_layout
):
    """An explicit `--layout`, or a detectable repo, never needs a terminal."""
    if repo_files is not None:
        monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(repo_files))

    result, captured = _invoke_add_readme(monkeypatch, extra_args, interactive=False)

    assert result.exit_code == 0, result.output
    assert captured["layout"] == expected_layout


@pytest.mark.unit
def test_invalid_layout_rejected(monkeypatch):
    """An unsupported `--layout` value is a usage error before generation."""
    result, captured = _invoke_add_readme(monkeypatch, ["--layout", "bogus"])

    assert result.exit_code != 0
    assert "Unsupported layout" in result.output
    assert captured == {}
