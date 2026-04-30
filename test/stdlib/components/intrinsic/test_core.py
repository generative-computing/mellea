"""Tests of the code in ``mellea.stdlib.intrinsics.core``"""

import gc
import json
import os
import pathlib

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

from mellea.backends.huggingface import LocalHFBackend
from mellea.core import ModelOutputThunk
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext
from test.conftest import cleanup_gpu_backend
from test.predicates import require_gpu
from test.stdlib.components.intrinsic.test_rag import (
    _read_input_json as _read_rag_input_json,
    _read_output_json as _read_rag_output_json,
)

# Skip entire module in CI since all tests are qualitative
pytestmark = [
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping core intrinsic tests in CI - all qualitative tests",
    ),
    pytest.mark.huggingface,
    require_gpu(min_vram_gb=12),
    pytest.mark.e2e,
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
    cleanup_gpu_backend(backend_, "test_core")


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
    return context, json_data


@pytest.mark.qualitative
def test_certainty(backend):
    """Verify that the uncertainty/certainty intrinsic functions properly."""
    context, _ = _read_input_json("uncertainty.json")

    result = core.check_certainty(context, backend)
    assert 0.0 <= result <= 1.0

    result2 = core.check_certainty(context, backend)
    assert 0.0 <= result2 <= 1.0


@pytest.mark.qualitative
def test_requirement_check(backend):
    """Verify that the requirement check intrinsic functions properly."""
    context, json_data = _read_input_json("requirement_check.json")
    requirement = json_data["requirement"]

    result = core.requirement_check(context, backend, requirement)
    assert 0.0 <= result <= 1.0

    result2 = core.requirement_check(context, backend, requirement)
    assert 0.0 <= result2 <= 1.0


@pytest.mark.xfail(
    strict=False,
    reason="Context attribution count varies non-deterministically across runs",
)
@pytest.mark.qualitative
def test_find_context_attributions(backend):
    """Verify that the context-attribution intrinsic functions properly."""
    context, assistant_response, documents = _read_rag_input_json(
        "context-attribution.json"
    )
    expected = _read_rag_output_json("context-attribution.json")

    result = core.find_context_attributions(
        assistant_response, documents, context, backend
    )
    # Even with temperature set to 0, there's some indeterminism with the the response.
    # Check only the initial responses for correctness.
    assert result[:7] == expected


@pytest.mark.xfail(
    strict=False,
    reason="Context attribution count varies non-deterministically across runs",
)
@pytest.mark.qualitative
def test_find_context_attributions_resolve(backend):
    """Verify context-attribution when response is resolved from context."""
    context, assistant_response, documents = _read_rag_input_json(
        "context-attribution.json"
    )
    context = context.add(ModelOutputThunk(value=assistant_response))
    expected = _read_rag_output_json("context-attribution.json")

    result = core.find_context_attributions(None, documents, context, backend)
    assert result[:7] == expected


if __name__ == "__main__":
    pytest.main([__file__])
