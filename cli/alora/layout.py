# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect the adapter layout of a Hugging Face Hub repository.

A trained adapter can be published in two layouts:

- **intrinsic**: weights live under `<intrinsic-name>/<base-model>/<lora|alora>/`,
  written by `m alora upload --intrinsic`. Mellea's `LocalFileBinding` and
  adapter-function loading look here.
- **flat**: weights sit at the repository root, written by plain
  `m alora upload`. PEFT can load this directly, but Mellea's intrinsic
  loading cannot find it.
"""

from collections.abc import Iterable, Mapping
from pathlib import PurePath
from typing import Any, Literal

RepoLayout = Literal["intrinsic", "flat", "unknown"]
AdapterTypeName = Literal["lora", "alora"]

_ADAPTER_TYPES: tuple[AdapterTypeName, ...] = ("lora", "alora")


def detect_repo_layout(repo_files: Iterable[str], intrinsic_name: str) -> RepoLayout:
    """Classify a repository's adapter layout from its file listing.

    This only classifies. It does not check that the weights are where the
    loader will look for them; see `find_adapter_types` for that.

    Args:
        repo_files: Paths returned by `HfApi.list_repo_files`, relative to the
            repository root.
        intrinsic_name: Adapter/intrinsic name — the leaf of the
            `<userid>/<intrinsic-name>` repository ID.

    Returns:
        `"intrinsic"` when any file is nested under `<intrinsic-name>/`, `"flat"`
        when an `adapter_config.json` sits at the repository root, and
        `"unknown"` otherwise (for example a repository containing only a README).
    """
    files = list(repo_files)
    prefix = f"{intrinsic_name}/"
    if any(path.startswith(prefix) for path in files):
        return "intrinsic"
    if "adapter_config.json" in files:
        return "flat"
    return "unknown"


def adapter_type_from_config(adapter_config: Mapping[str, Any]) -> AdapterTypeName:
    """Tell a LoRA adapter from an aLoRA adapter using its `adapter_config.json`.

    PEFT writes `alora_invocation_tokens` for every LoRA config and sets it to
    `null` unless the adapter is an activated LoRA, so the value decides, not
    whether the key is present.

    Args:
        adapter_config: Parsed contents of the adapter's `adapter_config.json`.

    Returns:
        `"alora"` if `alora_invocation_tokens` holds any tokens, else `"lora"`.
    """
    return "alora" if adapter_config.get("alora_invocation_tokens") else "lora"


def adapter_weight_dirs(
    intrinsic_name: str, base_model: str, repo_id: str
) -> dict[AdapterTypeName, str]:
    """Return where the loader looks for each adapter type's weights.

    Delegates to `adapter_subpath`, the same function the loader uses. Like
    `LocalHFBackend.base_model_name`, it looks the base model up by its last path
    component only, so `ibm-granite/granite-4.1-3b` and `granite-4.1-3b` resolve
    alike. That is also how `m alora upload --intrinsic` names the directory.

    Args:
        intrinsic_name: Adapter/intrinsic name — the leaf of the repository ID.
        base_model: Base model the adapter will be loaded against, as an ID or path.
        repo_id: Hugging Face Hub repository ID, which selects the directory layout.

    Returns:
        Mapping from adapter type to the repository-relative directory, without a
        trailing slash.
    """
    from mellea.formatters.granite.intrinsics.util import adapter_subpath

    base_model_name = PurePath(base_model).name
    return {
        adapter_type: adapter_subpath(
            intrinsic_name, base_model_name, repo_id, adapter_type=adapter_type
        )
        for adapter_type in _ADAPTER_TYPES
    }


def find_adapter_types(
    repo_files: Iterable[str], intrinsic_name: str, base_model: str, repo_id: str
) -> list[AdapterTypeName]:
    """List the adapter types whose weights the loader can find for a base model.

    Args:
        repo_files: Paths returned by `HfApi.list_repo_files`, relative to the
            repository root.
        intrinsic_name: Adapter/intrinsic name — the leaf of the repository ID.
        base_model: Base model the adapter will be loaded against.
        repo_id: Hugging Face Hub repository ID, which selects the directory layout.

    Returns:
        The adapter types, in `lora`, `alora` order, that have at least one file
        under the directory the loader resolves for `base_model`. Empty when the
        repository has nothing the loader would find for that base model, even if
        it holds weights for another.
    """
    files = list(repo_files)
    return [
        adapter_type
        for adapter_type, directory in adapter_weight_dirs(
            intrinsic_name, base_model, repo_id
        ).items()
        if any(path.startswith(f"{directory}/") for path in files)
    ]
