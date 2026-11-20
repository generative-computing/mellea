# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for what `m alora upload` tells the user about the layout it wrote.

The upload itself is replaced, so these exercise only the command's messaging
and which uploader it chooses.
"""

import json

import pytest
from typer.testing import CliRunner

pytest.importorskip("huggingface_hub", reason="huggingface_hub not installed")

from cli.m import cli

runner = CliRunner()


def test_flat_upload_warns_that_mellea_cannot_load_it_as_an_adapter_function(
    monkeypatch, tmp_path
):
    """A plain upload is allowed, but must not be silent about what it cannot do."""
    uploaded: dict = {}
    monkeypatch.setattr(
        "cli.alora.upload.upload_model", lambda **kwargs: uploaded.update(kwargs)
    )

    result = runner.invoke(
        cli, ["alora", "upload", str(tmp_path), "--name", "acme/stembolts"]
    )

    assert result.exit_code == 0, result.output
    assert uploaded == {"weight_path": str(tmp_path), "model_name": "acme/stembolts"}
    assert "flat layout" in result.output
    assert "--intrinsic --io-yaml" in result.output
    assert "Upload complete" in result.output


def test_intrinsic_upload_does_not_warn_about_the_flat_layout(monkeypatch, tmp_path):
    """The warning is only for the layout that cannot be loaded as an intrinsic."""
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "ibm-granite/granite-4.1-3b",
                "alora_invocation_tokens": [1, 2, 3],
            }
        )
    )
    io_yaml = tmp_path / "io.yaml"
    io_yaml.write_text("model: ~\n")
    uploaded: dict = {}
    monkeypatch.setattr(
        "cli.alora.intrinsic_uploader.upload_intrinsic",
        lambda **kwargs: uploaded.update(kwargs),
    )

    result = runner.invoke(
        cli,
        [
            "alora",
            "upload",
            str(tmp_path),
            "--name",
            "acme/stembolts",
            "--intrinsic",
            "--io-yaml",
            str(io_yaml),
        ],
    )

    assert result.exit_code == 0, result.output
    assert uploaded["model_name"] == "acme/stembolts"
    assert "flat layout" not in result.output
