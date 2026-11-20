# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adapter-layout detection helpers in `cli/alora/layout.py`."""

import pytest

from cli.alora.layout import (
    adapter_type_from_config,
    adapter_weight_dirs,
    detect_repo_layout,
    find_adapter_types,
)


@pytest.mark.parametrize(
    ("repo_files", "expected"),
    [
        # Layout written by `m alora upload --intrinsic`.
        (
            [
                "stembolts/granite-4.1-3b/alora/adapter_config.json",
                "stembolts/granite-4.1-3b/alora/adapter_model.safetensors",
            ],
            "intrinsic",
        ),
        # Classification only looks at the intrinsic's directory, not the subpath.
        (["stembolts/anything/else/adapter_config.json"], "intrinsic"),
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


def test_detect_repo_layout_prefers_intrinsic_over_root_weights():
    """A nested adapter wins when a repo also has (stale) root-level weights."""
    repo_files = [
        "adapter_config.json",
        "stembolts/granite-4.1-3b/alora/adapter_config.json",
    ]
    assert detect_repo_layout(repo_files, "stembolts") == "intrinsic"


def test_adapter_weight_dirs_matches_the_loader():
    """Directories come from `adapter_subpath`, canonicalizing the base model."""
    expected = {
        "lora": "stembolts/granite-4.1-3b/lora",
        "alora": "stembolts/granite-4.1-3b/alora",
    }
    # The org-qualified and bare spellings resolve to the same directory.
    for base_model in ("ibm-granite/granite-4.1-3b", "granite-4.1-3b"):
        assert (
            adapter_weight_dirs("stembolts", base_model, "acme/stembolts") == expected
        )


@pytest.mark.parametrize(
    ("repo_files", "expected"),
    [
        (["stembolts/granite-4.1-3b/alora/adapter_config.json"], ["alora"]),
        (["stembolts/granite-4.1-3b/lora/adapter_config.json"], ["lora"]),
        (
            [
                "stembolts/granite-4.1-3b/lora/adapter_config.json",
                "stembolts/granite-4.1-3b/alora/adapter_config.json",
            ],
            ["lora", "alora"],
        ),
        # Nested for a different base model: the loader would not find it.
        (["stembolts/granite-3.3-2b-instruct/alora/adapter_config.json"], []),
        # A directory that merely shares a prefix is not the adapter directory.
        (["stembolts/granite-4.1-3b/alora-old/adapter_config.json"], []),
        # Root-level weights are not under any directory the loader checks.
        (["adapter_config.json"], []),
        ([], []),
    ],
)
def test_find_adapter_types(repo_files, expected):
    """Only types with files under the loader's directory for the base model count."""
    assert (
        find_adapter_types(
            repo_files, "stembolts", "ibm-granite/granite-4.1-3b", "acme/stembolts"
        )
        == expected
    )


def test_find_adapter_types_uses_the_requested_base_model():
    """The same repository matches one base model and not another."""
    repo_files = ["stembolts/granite-3.3-2b-instruct/alora/adapter_config.json"]

    assert find_adapter_types(
        repo_files, "stembolts", "ibm-granite/granite-3.3-2b-instruct", "acme/stembolts"
    ) == ["alora"]
    assert (
        find_adapter_types(
            repo_files, "stembolts", "ibm-granite/granite-4.1-3b", "acme/stembolts"
        )
        == []
    )


@pytest.mark.parametrize(
    ("adapter_config", "expected"),
    [
        # PEFT writes the key for every LoRA config and nulls it unless activated,
        # so presence of the key must not be mistaken for an aLoRA.
        ({"alora_invocation_tokens": None, "r": 6}, "lora"),
        ({"alora_invocation_tokens": [], "r": 6}, "lora"),
        ({"r": 6}, "lora"),
        ({"alora_invocation_tokens": [49152, 2, 7], "r": 32}, "alora"),
    ],
)
def test_adapter_type_from_config(adapter_config, expected):
    """The tokens' value, not the key's presence, marks an activated LoRA."""
    assert adapter_type_from_config(adapter_config) == expected


@pytest.mark.parametrize(
    "base_model",
    ["myorg/my-finetune", "my-finetune", "/models/checkpoints/my-finetune"],
)
def test_adapter_weight_dirs_use_the_base_model_name_the_backend_uses(base_model):
    """A base model outside the canonical-name table resolves by its last path part.

    `LocalHFBackend.base_model_name` is `PurePath(model_id).name`, and
    `m alora upload --intrinsic` writes the same directory, so `myorg/my-finetune`
    is looked up as `my-finetune`. Passing the full ID to `adapter_subpath` would
    look in `stembolts/myorg/my-finetune/...` and report weights missing from a
    correctly uploaded adapter.
    """
    assert adapter_weight_dirs("stembolts", base_model, "acme/stembolts") == {
        "lora": "stembolts/my-finetune/lora",
        "alora": "stembolts/my-finetune/alora",
    }


def test_find_adapter_types_for_an_unmapped_org_qualified_base_model():
    """Weights uploaded for `myorg/my-finetune` are found for that same `--basemodel`."""
    repo_files = ["stembolts/my-finetune/lora/adapter_config.json"]

    assert find_adapter_types(
        repo_files, "stembolts", "myorg/my-finetune", "acme/stembolts"
    ) == ["lora"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_id",
    [
        "ibm-granite/granite-4.1-3b",
        "granite-4.1-3b",
        "myorg/my-finetune",
        "/models/checkpoints/my-finetune",
    ],
)
def test_adapter_weight_dirs_follow_the_real_backend_base_model_name(model_id):
    """The directory checked for `--basemodel` must be the one the backend loads.

    Calls the real `LocalHFBackend.base_model_name` property, so a change to how
    the backend names its base model fails here instead of silently making
    `add-readme` check the wrong directory.
    """
    from types import SimpleNamespace

    pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
    from mellea.backends.huggingface import LocalHFBackend

    # The property reads only `_model_id`, so no model needs loading.
    backend_name = LocalHFBackend.base_model_name.fget(  # type: ignore[attr-defined]
        SimpleNamespace(_model_id=model_id)
    )

    dirs = adapter_weight_dirs("stembolts", model_id, "acme/stembolts")
    assert dirs["alora"] == f"stembolts/{backend_name}/alora"
    assert dirs["lora"] == f"stembolts/{backend_name}/lora"
