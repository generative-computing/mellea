"""Tests of the code in ``mellea.stdlib.intrinsics.rag``"""

import gc
import json
import os
import pathlib

import pytest
import torch

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.base import ChatContext, Document
from mellea.stdlib.chat import Message
from mellea.stdlib.intrinsics import rag

DATA_ROOT = pathlib.Path(os.path.dirname(__file__)) / "testdata"
"""Location of data files for the tests in this file."""


BASE_MODEL = "ibm-granite/granite-3.3-2b-instruct"


@pytest.fixture(name="backend")
def _backend():
    """Backend used by the tests in this file."""

    # Prevent thrashing if the default device is CPU
    torch.set_num_threads(4)

    backend_ = LocalHFBackend(model_id=BASE_MODEL)
    yield backend_

    # Code after yield is cleanup code.
    # Free GPU memory with extreme prejudice.
    del backend_
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
        context = context.add(Message(m["role"], m["content"]))

    # Store the user turn at the end of the messages list separately so that tests can
    # play it back.
    next_user_turn = json_data["messages"][-1]["content"]

    documents = []
    if "extra_body" in json_data and "documents" in json_data["extra_body"]:
        for d in json_data["extra_body"]["documents"]:
            documents.append(Document(text=d["text"], doc_id=d["doc_id"]))
    return context, next_user_turn, documents


def _read_output_json(file_name: str):
    """Shared code for reading canned outputs stored in JSON files and converting
    to Mellea types."""
    with open(DATA_ROOT / "output_json" / file_name, encoding="utf-8") as f:
        json_data = json.load(f)

    # Output is in OpenAI chat completion response format. Assume only one choice.
    result_str = json_data["choices"][0]["message"]["content"]

    # Intrinsic outputs are always JSON, serialized to a string for OpenAI
    # compatibility.
    return json.loads(result_str)


@pytest.mark.qualitative
def test_answerability(backend):
    """Verify that the answerability intrinsic functions properly."""
    context, next_user_turn, documents = _read_input_json("answerability.json")

    # First call triggers adapter loading
    result = rag.check_answerability(next_user_turn, documents, context, backend)
    assert pytest.approx(result) == 1.0

    # Second call hits a different code path from the first one
    result = rag.check_answerability(next_user_turn, documents, context, backend)
    assert pytest.approx(result) == 1.0


@pytest.mark.qualitative
def test_query_rewrite(backend):
    """Verify that the answerability intrinsic functions properly."""
    context, next_user_turn, _ = _read_input_json("query_rewrite.json")
    expected = (
        "Is Rex, the dog, more likely to get fleas because he spends a lot of "
        "time outdoors?"
    )

    # First call triggers adapter loading
    result = rag.rewrite_question(next_user_turn, context, backend)
    assert result == expected

    # Second call hits a different code path from the first one
    result = rag.rewrite_question(next_user_turn, context, backend)
    assert result == expected


@pytest.mark.qualitative
def test_citations(backend):
    """Verify that the citations intrinsic functions properly."""
    context, assistant_response, docs = _read_input_json("citations.json")
    expected = _read_output_json("citations.json")

    # First call triggers adapter loading
    result = rag.find_citations(assistant_response, docs, context, backend)
    assert result == expected

    # Second call hits a different code path from the first one
    result = rag.find_citations(assistant_response, docs, context, backend)
    assert result == expected


@pytest.mark.qualitative
def test_context_relevance(backend):
    """Verify that the context relevance intrinsic functions properly."""
    context, question, docs = _read_input_json("context_relevance.json")

    # Context relevance can only check against a single document at a time.
    document = docs[0]

    # First call triggers adapter loading
    result = rag.check_context_relevance(question, document, context, backend)
    assert pytest.approx(result, abs=2e-2) == 0.45

    # Second call hits a different code path from the first one
    result = rag.check_context_relevance(question, document, context, backend)
    assert pytest.approx(result, abs=2e-2) == 0.45


@pytest.mark.qualitative
def test_hallucination_detection(backend):
    """Verify that the hallucination detection intrinsic functions properly."""
    context, assistant_response, docs = _read_input_json("hallucination_detection.json")
    expected = _read_output_json("hallucination_detection.json")

    # First call triggers adapter loading
    result = rag.flag_hallucinated_content(assistant_response, docs, context, backend)
    # pytest.approx() chokes on lists of records, so we do this complicated dance.
    for r, e in zip(result, expected, strict=True):
        assert pytest.approx(r, abs=2e-2) == e

    # Second call hits a different code path from the first one
    result = rag.flag_hallucinated_content(assistant_response, docs, context, backend)
    for r, e in zip(result, expected, strict=True):
        assert pytest.approx(r, abs=2e-2) == e


@pytest.mark.qualitative
def test_answer_relevance(backend):
    """Verify that the answer relevance composite intrinsic functions properly."""
    context, answer, docs = _read_input_json("answer_relevance.json")

    # Note that this is not the optimal answer. This test is currently using an
    # outdated LoRA adapter. Releases of new adapters will come after the Mellea
    # integration has stabilized.
    expected_rewrite = (
        "The documents do not provide information about the attendees of the meeting."
    )

    # First call triggers adapter loading
    result = rag.rewrite_answer_for_relevance(answer, docs, context, backend)
    assert result == expected_rewrite

    # Second call hits a different code path from the first one
    result = rag.rewrite_answer_for_relevance(answer, docs, context, backend)
    assert result == expected_rewrite

    # Canned input always gets rewritten. Set threshold to disable the rewrite.
    result = rag.rewrite_answer_for_relevance(
        answer, docs, context, backend, rewrite_threshold=0.0
    )
    assert result == answer


@pytest.mark.qualitative
def test_answer_relevance_classifier(backend):
    """Verify that the first phase of the answer relevance flow behaves as expectee."""
    context, answer, docs = _read_input_json("answer_relevance.json")

    result_json = rag._call_intrinsic(
        "answer_relevance_classifier",
        context.add(Message("assistant", answer, documents=list(docs))),
        backend,
    )
    assert result_json["answer_relevance_likelihood"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
