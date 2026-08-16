# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mellea."""

from importlib.metadata import PackageNotFoundError, version

from . import serve
from .backends import model_ids
from .stdlib.components.genstub import generative
from .stdlib.session import MelleaSession, start_session
from .stdlib.start_backend import start_backend

try:
    # Read the version from the installed package metadata so it stays in sync
    # with the `version` field in pyproject.toml (no manual duplication).
    __version__ = version("mellea")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "MelleaSession",
    "__version__",
    "generative",
    "model_ids",
    "serve",
    "start_backend",
    "start_session",
]
