# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free types shared by the decompose CLI and pipeline.

Lives in its own module so that `m decompose --help` can build its Typer
signature without importing `cli.decompose.pipeline`, which pulls in `mellea`
and its backend dependencies. Re-exported from `cli.decompose.pipeline` for
backwards compatibility.
"""

from enum import StrEnum


class DecompBackend(StrEnum):
    """Inference backends supported by the decomposition pipeline.

    Attributes:
        ollama: Local Ollama inference server backend.
        openai: OpenAI-compatible HTTP API backend.
    """

    ollama = "ollama"
    openai = "openai"
