# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for embedded adapter functions on LocalHFBackend.

Uses the project's standard 3B Granite Switch checkpoint. It runs in normal
local pytest on supported GPU hardware and is skipped in CI.
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
    require_gpu(min_vram_gb=12),
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping local Granite Switch e2e test in CI",
    ),
]

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext


@pytest.fixture(scope="module")
def backend() -> Iterator[LocalHFBackend]:
    """Load a Granite Switch checkpoint with embedded adapters registered."""
    with hf_skip():
        backend = LocalHFBackend(
            model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW, load_embedded_adapters=True
        )
    yield backend
    cleanup_gpu_backend(backend, "huggingface")


def test_embedded_answerability_runs_on_local_hf(backend: LocalHFBackend) -> None:
    """The local template renders an embedded control token and parses output."""
    messages = [{"role": "user", "content": "Can these documents answer my question?"}]
    base_prompt = backend._tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    adapter_prompt = backend._tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        adapter_name="answerability",
        tokenize=False,
    )
    assert adapter_prompt != base_prompt, (
        "Granite Switch ignored adapter_name; the embedded adapter control token "
        "was not rendered."
    )

    result = rag.check_answerability(
        "What is the square root of 4?",
        ["The square root of 4 is 2."],
        ChatContext().add(Message("user", messages[0]["content"])),
        backend,
    )

    assert result in {"answerable", "unanswerable"}
