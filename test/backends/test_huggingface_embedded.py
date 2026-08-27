# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for embedded adapter functions on LocalHFBackend.

Set `GRANITE_SWITCH_MODEL_ID` to a local-HF-compatible Granite Switch
checkpoint before running this module. It is skipped by default because the
checkpoint requires a GPU and is not part of normal CI.
"""

import os
from collections.abc import Iterator

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)

from test.conftest import cleanup_gpu_backend, hf_skip
from test.predicates import require_gpu

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.huggingface,
    pytest.mark.slow,
    require_gpu(min_vram_gb=12),
    pytest.mark.skipif(
        not os.environ.get("GRANITE_SWITCH_MODEL_ID"),
        reason="Set GRANITE_SWITCH_MODEL_ID to run LocalHF embedded-adapter tests",
    ),
]

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext


@pytest.fixture(scope="module")
def backend() -> Iterator[LocalHFBackend]:
    """Load a Granite Switch checkpoint with embedded adapters registered."""
    with hf_skip():
        backend = LocalHFBackend(
            model_id=os.environ["GRANITE_SWITCH_MODEL_ID"], load_embedded_adapters=True
        )
    yield backend
    cleanup_gpu_backend(backend, "huggingface")


def test_embedded_answerability_runs_on_local_hf(backend: LocalHFBackend) -> None:
    """The local HF path renders the embedded adapter control token and parses output."""
    result = rag.check_answerability(
        "What is the square root of 4?",
        ["The square root of 4 is 2."],
        ChatContext().add(Message("user", "Can these documents answer my question?")),
        backend,
    )

    assert result in {"answerable", "unanswerable"}
