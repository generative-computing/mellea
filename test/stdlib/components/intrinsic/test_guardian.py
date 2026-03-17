"""Tests of the code in ``mellea.stdlib.intrinsics.core``"""

import gc
import json
import os
import pathlib

import pytest
import torch

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message, Document
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext

# Skip entire module in CI since all tests are qualitative
pytestmark = [
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping core intrinsic tests in CI - all qualitative tests",
    ),
    pytest.mark.huggingface,
    pytest.mark.requires_gpu,
    pytest.mark.requires_heavy_ram,
    pytest.mark.llm,
]

DATA_ROOT = pathlib.Path(os.path.dirname(__file__)) / "testdata"
"""Location of data files for the tests in this file."""


BASE_MODEL = "ibm-granite/granite-4.0-micro"


@pytest.fixture(name="backend", scope="module")
def _backend():
    """Backend used by the tests in this file. Module-scoped to avoid reloading the 3B model for each test."""
    # Prevent thrashing if the default device is CPU
    torch.set_num_threads(4)

    backend_ = LocalHFBackend(model_id=BASE_MODEL)
    yield backend_

    # Code after yield is cleanup code.
    # Free GPU memory with extreme prejudice.
    del backend_
    gc.collect()
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()


def _read_input_json(file_name: str):
    """Read test data from JSON and convert to a ChatContext.

    Returns the context and the raw JSON data (for accessing extra fields
    like ``requirement``).
    """
    with open(DATA_ROOT / "input_json" / file_name, encoding="utf-8") as f:
        json_data = json.load(f)

    context = ChatContext()
    for m in json_data["messages"]:
        context = context.add(Message(m["role"], m["content"]))

    if "extra_body" in json_data and "documents" in json_data["extra_body"]:
        for d in json_data["extra_body"]["documents"]:
            context.add(Document(text=d["text"], doc_id=d["doc_id"]))
    return context, json_data

@pytest.mark.qualitative
def test_factuality_detection(backend):
    """Verify that the factuality detection intrinsic functions properly."""
    context, _ = _read_input_json("factuality_detection.json")

    result = guardian.factuality_detection(context, backend)
    assert result == "yes" or result == "no"

@pytest.mark.qualitative
def test_factuality_correction(backend):
    """Verify that the factuality correction intrinsic functions properly."""
    context, _ = _read_input_json("factuality_correction.json")

    result = guardian.factuality_correction(context, backend)
    assert isinstance(result, str)

if __name__ == "__main__":
    pytest.main([__file__])
