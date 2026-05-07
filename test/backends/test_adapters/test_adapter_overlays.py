"""Tests for the in-repo ``io.yaml`` overlay mechanism in ``IntrinsicAdapter``.

Overlays let Mellea ship ``io.yaml`` content ahead of an upstream release; the
adapter loader prefers them over the Hugging Face cache when present. These
tests exercise the resolution helper and the adapter's use of it, using only
the overlay files that ship with the repo (no network access).
"""

import importlib.resources
import pathlib
from unittest.mock import patch

import pytest

from mellea.backends.adapters import IntrinsicAdapter
from mellea.backends.adapters.adapter import _find_overlay_io_yaml
from mellea.backends.adapters.catalog import AdapterType

GUARDIAN_INTRINSICS = [
    "guardian-core",
    "policy-guardrails",
    "factuality-detection",
    "factuality-correction",
]


@pytest.mark.parametrize("intrinsic_name", GUARDIAN_INTRINSICS)
def test_overlay_resolves_for_granite_4_1_3b_lora(intrinsic_name):
    """Each Guardian intrinsic has a lora overlay for granite-4.1-3b."""
    path = _find_overlay_io_yaml(
        intrinsic_name, "ibm-granite/granite-4.1-3b", alora=False
    )
    assert path is not None
    assert path.is_file()
    assert path.name == "io.yaml"
    # Overlay dir matches the HF layout: <intrinsic>/<model>/<lora|alora>
    assert path.parent.name == "lora"
    assert path.parent.parent.name == "granite-4.1-3b"
    assert path.parent.parent.parent.name == intrinsic_name


@pytest.mark.parametrize("intrinsic_name", GUARDIAN_INTRINSICS)
def test_overlay_resolves_with_canonical_short_name(intrinsic_name):
    """Passing the canonical short model name finds the same overlay."""
    long_form = _find_overlay_io_yaml(
        intrinsic_name, "ibm-granite/granite-4.1-3b", alora=False
    )
    short_form = _find_overlay_io_yaml(intrinsic_name, "granite-4.1-3b", alora=False)
    assert long_form == short_form


def test_overlay_returns_none_for_unknown_intrinsic():
    """Helper returns None when no overlay is present."""
    path = _find_overlay_io_yaml(
        "answerability", "ibm-granite/granite-4.1-3b", alora=False
    )
    assert path is None


def test_overlay_returns_none_for_unknown_model():
    """Helper returns None when the intrinsic has no overlay for the given model."""
    # guardian-core exists, but not against a made-up model name
    path = _find_overlay_io_yaml(
        "guardian-core", "granite-nonexistent-model", alora=False
    )
    assert path is None


def test_overlay_distinguishes_lora_and_alora():
    """lora and alora overlays resolve to different files where upstream differs.

    factuality-correction on granite-4.1-3b has an alora variant that differs
    from the lora variant (a comment line is missing). Confirm the resolver
    picks up that distinction rather than collapsing both to the same file.
    """
    lora_path = _find_overlay_io_yaml(
        "factuality-correction", "granite-4.1-3b", alora=False
    )
    alora_path = _find_overlay_io_yaml(
        "factuality-correction", "granite-4.1-3b", alora=True
    )
    assert lora_path is not None
    assert alora_path is not None
    assert lora_path != alora_path
    assert lora_path.read_bytes() != alora_path.read_bytes()


@pytest.mark.parametrize("intrinsic_name", GUARDIAN_INTRINSICS)
def test_intrinsic_adapter_loads_overlay_without_hitting_hf(intrinsic_name):
    """IntrinsicAdapter uses the overlay and never calls obtain_io_yaml."""
    with patch(
        "mellea.backends.adapters.adapter.intrinsics.obtain_io_yaml"
    ) as mock_obtain:
        # Force a failure if anything tries to fall back to HF.
        mock_obtain.side_effect = AssertionError(
            "obtain_io_yaml should not be called when an overlay is present"
        )
        adapter = IntrinsicAdapter(
            intrinsic_name,
            adapter_type=AdapterType.LORA,
            base_model_name="ibm-granite/granite-4.1-3b",
        )
    assert adapter.config is not None
    # Sanity: every Guardian io.yaml today carries the intrinsic's name.
    assert adapter.config.get("name", "").strip() == intrinsic_name
    mock_obtain.assert_not_called()


def test_policy_guardrails_micro_overlay_preserves_label_variant():
    """The granite-4.0-micro policy-guardrails overlay uses ``label`` (not ``score``).

    This preserves upstream's current drift. When upstream converges on a single
    schema, the overlay can be updated to match.
    """
    path = _find_overlay_io_yaml("policy-guardrails", "granite-4.0-micro", alora=False)
    assert path is not None
    content = path.read_text(encoding="utf-8")
    assert '"label"' in content
    assert '"score"' not in content


def test_non_overlaid_intrinsic_still_falls_back_to_hf():
    """Intrinsics without an overlay go through obtain_io_yaml as before."""
    sentinel_path = pathlib.Path("/tmp/sentinel-io.yaml")

    def fake_obtain_io_yaml(intrinsic_name, base_model_name, repo_id, alora=False):
        # Write a minimal valid config to the sentinel path and return it.
        sentinel_path.write_text(
            "name: sentinel\nmodel: ~\nresponse_format: '{}'\ntransformations: ~\n"
        )
        return sentinel_path

    with patch(
        "mellea.backends.adapters.adapter.intrinsics.obtain_io_yaml",
        side_effect=fake_obtain_io_yaml,
    ) as mock_obtain:
        adapter = IntrinsicAdapter(
            "answerability",
            adapter_type=AdapterType.LORA,
            base_model_name="ibm-granite/granite-4.1-3b",
        )
    assert mock_obtain.called
    assert adapter.config["name"] == "sentinel"


def test_overlay_yaml_files_are_package_data():
    """The overlay io.yaml files must ship in the wheel.

    Uses importlib.resources to verify the files are discoverable through the
    package interface, which is what a wheel install exposes.
    """
    overlays = importlib.resources.files("mellea.backends.adapters") / "_overlays"
    for intrinsic_name in GUARDIAN_INTRINSICS:
        candidate = overlays / intrinsic_name / "granite-4.1-3b" / "lora" / "io.yaml"
        assert candidate.is_file(), f"missing package-data overlay: {candidate}"
