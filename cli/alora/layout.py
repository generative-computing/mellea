# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect the adapter layout of a Hugging Face Hub repository.

A trained adapter can be published in two layouts:

- **intrinsic**: weights live under ``<intrinsic-name>/<base-model>/<lora|alora>/``,
  written by ``m alora upload --intrinsic``. Mellea's ``LocalFileBinding`` and
  adapter-function loading look here.
- **flat**: weights sit at the repository root, written by plain
  ``m alora upload``. PEFT can load this directly, but Mellea's intrinsic
  loading cannot find it.
"""

from collections.abc import Iterable
from typing import Literal

RepoLayout = Literal["intrinsic", "flat", "unknown"]


def detect_repo_layout(repo_files: Iterable[str], intrinsic_name: str) -> RepoLayout:
    """Classify a repository's adapter layout from its file listing.

    Args:
        repo_files: Paths returned by ``HfApi.list_repo_files``, relative to the
            repository root.
        intrinsic_name: Adapter/intrinsic name — the leaf of the
            ``<userid>/<intrinsic-name>`` repository ID.

    Returns:
        ``"intrinsic"`` when any file is nested under ``<intrinsic-name>/`` (this
        also covers the older ``<intrinsic-name>/<lora|alora>/<base-model>/``
        layout), ``"flat"`` when an ``adapter_config.json`` sits at the repository
        root, and ``"unknown"`` otherwise (for example a repository containing
        only a README).
    """
    files = list(repo_files)
    prefix = f"{intrinsic_name}/"
    if any(path.startswith(prefix) for path in files):
        return "intrinsic"
    if "adapter_config.json" in files:
        return "flat"
    return "unknown"
