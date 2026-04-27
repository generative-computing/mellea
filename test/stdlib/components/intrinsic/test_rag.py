"""Tests of the code in ``mellea.stdlib.intrinsics.rag``"""

import gc
import json
import os
import pathlib

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B
from mellea.core import ModelOutputThunk
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext
from test.predicates import require_gpu

# Skip entire module in CI since all 7 tests are qualitative
pytestmark = [
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping RAG tests in CI - all qualitative tests",
    ),
    pytest.mark.huggingface,
    require_gpu(min_vram_gb=12),
    pytest.mark.e2e,
]

DATA_ROOT = pathlib.Path(os.path.dirname(__file__)) / "testdata"
"""Location of data files for the tests in this file."""


@pytest.fixture(name="backend", scope="module")
def _backend():
    """Backend used by the tests in this file. Module-scoped to avoid reloading the 3B model for each test."""
    # Prevent thrashing if the default device is CPU
    torch.set_num_threads(4)

    # No adapters for hybrid version.
    backend_ = LocalHFBackend(model_id=IBM_GRANITE_4_MICRO_3B.hf_model_name)  # type: ignore
    yield backend_

    from test.conftest import cleanup_gpu_backend

    cleanup_gpu_backend(backend_, "rag")


def _read_input_json(file_name: str):
    """Shared code for reading data stored in JSON files and converting to Mellea
    types.
    """
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
    to Mellea types.
    """
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
    assert result == "answerable"

    # Second call hits a different code path from the first one
    result = rag.check_answerability(next_user_turn, documents, context, backend)
    assert result == "answerable"


@pytest.mark.qualitative
def test_query_rewrite(backend):
    """Verify that the answerability intrinsic functions properly."""
    context, next_user_turn, _ = _read_input_json("query_rewrite.json")
    expected = (
        "Is Rex more likely to get fleas because he spends a lot of time outdoors?"
    )

    # First call triggers adapter loading
    result = rag.rewrite_question(next_user_turn, context, backend)
    assert result == expected

    # Second call hits a different code path from the first one
    result = rag.rewrite_question(next_user_turn, context, backend)
    assert result == expected


@pytest.mark.xfail(reason="Non-deterministic citation boundaries across environments")
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
    assert result == "irrelevant"

    # Second call hits a different code path from the first one
    result = rag.check_context_relevance(question, document, context, backend)
    assert result == "irrelevant"


@pytest.mark.qualitative
def test_hallucination_detection(backend):
    """Verify that the hallucination detection intrinsic functions properly."""
    context, assistant_response, docs = _read_input_json("hallucination_detection.json")
    expected = _read_output_json("hallucination_detection.json")

    # First call triggers adapter loading
    result = rag.flag_hallucinated_content(assistant_response, docs, context, backend)
    # pytest.approx() chokes on lists of records, so we do this complicated dance.
    for r, e in zip(result, expected, strict=True):  # type: ignore
        assert r == e

    # Second call hits a different code path from the first one
    result = rag.flag_hallucinated_content(assistant_response, docs, context, backend)
    for r, e in zip(result, expected, strict=True):  # type: ignore
        assert r == e


@pytest.mark.qualitative
def test_query_clarification_positive(backend):
    """Verify that query clarification detects ambiguous queries requiring clarification."""
    context, next_user_turn, documents = _read_input_json(
        "query_clarification_positive.json"
    )

    # First call triggers adapter loading
    result = rag.clarify_query(next_user_turn, documents, context, backend)
    # The result should be a clarification question, not "CLEAR"
    assert result != "CLEAR"
    assert len(result) > 0

    # Second call hits a different code path from the first one
    result = rag.clarify_query(next_user_turn, documents, context, backend)
    assert result != "CLEAR"
    assert len(result) > 0


@pytest.mark.qualitative
def test_query_clarification_negative(backend):
    """Verify that query clarification returns CLEAR for clear queries."""
    context, next_user_turn, documents = _read_input_json(
        "query_clarification_negative.json"
    )

    # First call triggers adapter loading
    result = rag.clarify_query(next_user_turn, documents, context, backend)
    # The result should be "CLEAR" for a clear query that doesn't need clarification
    assert result == "CLEAR"

    # Second call hits a different code path from the first one
    result = rag.clarify_query(next_user_turn, documents, context, backend)
    assert result == "CLEAR"


# ---------------------------------------------------------------------------
# Resolve-from-context variants: pass question/response=None, infer from ctx
# ---------------------------------------------------------------------------


@pytest.mark.qualitative
def test_answerability_resolve(backend):
    """Verify answerability when question is resolved from context."""
    context, next_user_turn, documents = _read_input_json("answerability.json")
    context = context.add(Message("user", next_user_turn))

    result = rag.check_answerability(None, documents, context, backend)
    assert result == "answerable"


@pytest.mark.qualitative
def test_query_rewrite_resolve(backend):
    """Verify query rewrite when question is resolved from context."""
    context, next_user_turn, _ = _read_input_json("query_rewrite.json")
    context = context.add(Message("user", next_user_turn))
    expected = (
        "Is Rex more likely to get fleas because he spends a lot of time outdoors?"
    )

    result = rag.rewrite_question(None, context, backend)
    assert result == expected


@pytest.mark.xfail(reason="Non-deterministic citation boundaries across environments")
@pytest.mark.qualitative
def test_citations_resolve(backend):
    """Verify citations when response is resolved from context."""
    context, assistant_response, docs = _read_input_json("citations.json")
    context = context.add(ModelOutputThunk(value=assistant_response))
    expected = _read_output_json("citations.json")

    result = rag.find_citations(None, docs, context, backend)
    assert result == expected


@pytest.mark.qualitative
def test_context_relevance_resolve(backend):
    """Verify context relevance when question is resolved from context."""
    context, question, docs = _read_input_json("context_relevance.json")
    context = context.add(Message("user", question))
    document = docs[0]

    result = rag.check_context_relevance(None, document, context, backend)
    assert result == "irrelevant"


@pytest.mark.qualitative
def test_hallucination_detection_resolve(backend):
    """Verify hallucination detection when response is resolved from context."""
    context, assistant_response, docs = _read_input_json("hallucination_detection.json")
    context = context.add(ModelOutputThunk(value=assistant_response))
    expected = _read_output_json("hallucination_detection.json")

    result = rag.flag_hallucinated_content(None, docs, context, backend)
    for r, e in zip(result, expected, strict=True):  # type: ignore
        assert r == e


@pytest.mark.qualitative
def test_query_clarification_positive_resolve(backend):
    """Verify query clarification (positive) when question is resolved from context."""
    context, next_user_turn, documents = _read_input_json(
        "query_clarification_positive.json"
    )
    context = context.add(Message("user", next_user_turn))

    result = rag.clarify_query(None, documents, context, backend)
    assert result != "CLEAR"
    assert len(result) > 0


@pytest.mark.qualitative
def test_query_clarification_negative_resolve(backend):
    """Verify query clarification (negative) when question is resolved from context."""
    context, next_user_turn, documents = _read_input_json(
        "query_clarification_negative.json"
    )
    context = context.add(Message("user", next_user_turn))

    result = rag.clarify_query(None, documents, context, backend)
    assert result == "CLEAR"


if __name__ == "__main__":
    pytest.main([__file__])
