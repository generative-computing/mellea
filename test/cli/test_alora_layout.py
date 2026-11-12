# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adapter-layout detection helpers in `cli/alora/layout.py`."""

import pytest

from cli.alora.layout import detect_repo_layout


@pytest.mark.unit
@pytest.mark.parametrize(
    ("repo_files", "expected"),
    [
        # Current intrinsic layout: <intrinsic>/<base-model>/<adapter-type>/...
        (
            [
                "stembolts/granite-4.1-3b/alora/adapter_config.json",
                "stembolts/granite-4.1-3b/alora/adapter_model.safetensors",
            ],
            "intrinsic",
        ),
        # Older intrinsic layout: <intrinsic>/<adapter-type>/<base-model>/...
        (["stembolts/alora/granite-4.1-3b/adapter_config.json"], "intrinsic"),
        # Plain `m alora upload`: weights at the repository root.
        (["adapter_config.json", "adapter_model.safetensors", "README.md"], "flat"),
        # A repository with only a README cannot be classified.
        (["README.md"], "unknown"),
        ([], "unknown"),
        # A different adapter's directory does not count as this one's layout.
        (["other-adapter/granite-4.1-3b/alora/adapter_config.json"], "unknown"),
    ],
)
def test_detect_repo_layout(repo_files, expected):
    """`detect_repo_layout` classifies intrinsic, flat, and unrecognized repos."""
    assert detect_repo_layout(repo_files, "stembolts") == expected


@pytest.mark.unit
def test_detect_repo_layout_prefers_intrinsic_over_root_weights():
    """A nested adapter wins when a repo also has (stale) root-level weights."""
    repo_files = [
        "adapter_config.json",
        "stembolts/granite-4.1-3b/alora/adapter_config.json",
    ]
    assert detect_repo_layout(repo_files, "stembolts") == "intrinsic"
