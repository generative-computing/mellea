"""Tests of the code in ``mellea.stdlib.intrinsics.rag``"""

import gc
import os
import json
import pathlib

import pytest
import torch

from mellea.backends.formatter import TemplateFormatter
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.base import ChatContext, Document
from mellea.stdlib.chat import Message
from mellea.stdlib.intrinsics import rag


DATA_ROOT = pathlib.Path(os.path.dirname(__file__)) / "testdata"
"""Location of data files for the tests in this file."""

BASE_MODEL = "ibm-granite/granite-3.3-2b-instruct"


@pytest.fixture
def backend():
    """Backend used by the tests in this file."""

    backend = LocalHFBackend(
        model_id=BASE_MODEL, formatter=TemplateFormatter(model_id=BASE_MODEL)
    )
    yield backend

    # Begin cleanup code
    del backend
    gc.collect()  # Force a collection of the newest generation
    gc.collect()
    gc.collect()  # Hopefully force a collection of the oldest generation
    torch.cuda.empty_cache()


def _read_input_json(file_name: str):
    """Shared code for reading data stored in JSON files and converting to Mellea
    types."""
    with open(DATA_ROOT / "input_json" / file_name, encoding="utf-8") as f:
        json_data = json.load(f)

    # Data is assumed to be an OpenAI chat completion request. Convert to Mellea format.
    context = ChatContext()
    for m in json_data["messages"][:-1]:
        context.add(Message(m["role"], m["content"]))

    # Store the user turn at the end of the messages list separately so that tests can
    # play it back.
    next_user_turn = json_data["messages"][-1]["content"]

    documents = []
    if "extra_body" in json_data and "documents" in json_data["extra_body"]:
        for d in json_data["extra_body"]["documents"]:
            documents.append(
                Document(
                    text=d["text"],
                    # Mellea doesn't have a doc_id, but OpenAI requires it. Compromise
                    # by converting the doc_id to a title.
                    title=d["doc_id"],
                )
            )
    return context, next_user_turn, documents


def test_answerability(backend):
    """Verify that the answerability intrinsic functions properly."""
    context, next_user_turn, documents = _read_input_json("answerability.json")

    # First call triggers LoRA adapter loading
    result = rag.check_answerability(context, next_user_turn, documents, backend)
    assert pytest.approx(result) == 1.0

    # Second call hits a different code path from the first one
    result = rag.check_answerability(context, next_user_turn, documents, backend)
    assert pytest.approx(result) == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
