# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `m alora add-readme` layout detection wiring.

The README generator's LLM and the Hugging Face upload are replaced so these
tests exercise only the layout resolution in `cli/alora/commands.py`.
"""

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


def _invoke_add_readme(monkeypatch, extra_args: list[str]):
    """Run `add-readme` with generation and uploading stubbed out.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        extra_args: Additional CLI arguments appended after `--name`.

    Returns:
        Tuple of the CLI result and the kwargs passed to `generate_readme`.
    """
    captured: dict = {}

    def fake_generate_readme(**kwargs):
        captured.update(kwargs)
        with open(kwargs["output_path"], "w") as fh:
            fh.write("# README\n")
        return kwargs["output_path"]

    monkeypatch.setattr("huggingface_hub.get_token", lambda: "test-token")
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
def test_invalid_layout_rejected(monkeypatch):
    """An unsupported `--layout` value is a usage error before generation."""
    result, captured = _invoke_add_readme(monkeypatch, ["--layout", "bogus"])

    assert result.exit_code != 0
    assert "Unsupported layout" in result.output
    assert captured == {}
