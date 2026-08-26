# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mellea."""

# Concrete backends are deliberately not re-exported here. They have optional
# dependencies and are imported explicitly from `mellea.backends.<provider>` so
# that a missing extra produces a targeted install hint rather than an error on
# `import mellea`.

from importlib.metadata import PackageNotFoundError, version

from . import serve
from .backends import model_ids
from .backends.model_options import ModelOption
from .core import (
    Backend,
    CBlock,
    Component,
    Context,
    MelleaLogger,
    ModelOutputThunk,
    Requirement,
    SamplingResult,
    TemplateRepresentation,
    ValidationResult,
)
from .stdlib import functional as mfuncs
from .stdlib.components import (
    Document,
    Instruction,
    Intrinsic,
    Message,
    SimpleComponent,
    mify,
)
from .stdlib.components.genstub import generative
from .stdlib.context import ChatContext, SimpleContext
from .stdlib.requirements import check, req, simple_validate
from .stdlib.sampling import RejectionSamplingStrategy
from .stdlib.session import MelleaSession, start_session
from .stdlib.start_backend import start_backend

try:
    # Read the version from the installed package metadata so it stays in sync
    # with the `version` field in pyproject.toml (no manual duplication).
    __version__ = version("mellea")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Backend",
    "CBlock",
    "ChatContext",
    "Component",
    "Context",
    "Document",
    "Instruction",
    "Intrinsic",
    "MelleaLogger",
    "MelleaSession",
    "Message",
    "ModelOption",
    "ModelOutputThunk",
    "RejectionSamplingStrategy",
    "Requirement",
    "SamplingResult",
    "SimpleComponent",
    "SimpleContext",
    "TemplateRepresentation",
    "ValidationResult",
    "__version__",
    "check",
    "generative",
    "mfuncs",
    "mify",
    "model_ids",
    "req",
    "serve",
    "simple_validate",
    "start_backend",
    "start_session",
]
