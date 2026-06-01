"""End-to-end tests for GroundednessRequirement with simple use cases."""

import pytest

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.rag import GroundednessRequirement
from test.predicates import require_gpu


@pytest.fixture(scope="session")
def backend():
    """Provide HuggingFace backend for tests (session-scoped to avoid OOM)."""
    return LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")


@pytest.fixture
def simple_docs():
    """Provide simple documents for basic validation."""
    return [
        Document(
            doc_id="0",
            text="Paris is the capital of France. It is located in northern France.",
        ),
        Document(
            doc_id="1",
            text="London is the capital of England. It is located on the River Thames.",
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
async def test_groundedness_e2e_fully_grounded(backend, simple_docs):
    """End-to-end test: response fully grounded by documents."""
    req = GroundednessRequirement(documents=simple_docs, allow_partial_support=False)

    ctx = (
        ChatContext()
        .add(Message("user", "What is the capital of France?"))
        .add(Message("assistant", "Paris is the capital of France."))
    )

    result = await req.validate(backend, ctx)
    # Response is fully grounded in the provided documents
    assert result.as_bool() is True
    assert result.reason is not None


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.huggingface
@pytest.mark.qualitative
@require_gpu(min_vram_gb=8)
async def test_groundedness_e2e_ungrounded_claim(backend, simple_docs):
    """End-to-end test: response contains ungrounded claim."""
    req = GroundednessRequirement(documents=simple_docs, allow_partial_support=False)

    ctx = (
        ChatContext()
        .add(Message("user", "Tell me about European capitals"))
        .add(
            Message(
                "assistant",
                "Paris is the capital of France. Berlin is the capital of Germany.",
            )
        )
    )

    result = await req.validate(backend, ctx)
    # Berlin claim is not grounded (not in documents), should return False
    assert result.as_bool() is False
    assert result.reason is not None


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
async def test_groundedness_e2e_documents_in_message(backend, simple_docs):
    """End-to-end test: documents provided in message instead of constructor."""
    req = GroundednessRequirement(allow_partial_support=False)

    ctx = (
        ChatContext()
        .add(Message("user", "What is the capital of France?"))
        .add(
            Message(
                "assistant", "Paris is the capital of France.", documents=simple_docs
            )
        )
    )

    result = await req.validate(backend, ctx)
    # Response is grounded in the provided documents
    assert result.as_bool() is True


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
async def test_groundedness_e2e_multi_message_conversation(backend, simple_docs):
    """End-to-end test: grounded response in multi-message conversation."""
    req = GroundednessRequirement(documents=simple_docs, allow_partial_support=False)

    ctx = (
        ChatContext()
        .add(Message("user", "What is the capital of France?"))
        .add(Message("assistant", "Paris is the capital of France."))
        .add(Message("user", "Where is Paris located?"))
        .add(Message("assistant", "Paris is located in northern France."))
    )

    result = await req.validate(backend, ctx)
    # Only the last assistant message is validated, and it is grounded
    assert result.as_bool() is True


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
async def test_groundedness_e2e_allow_partial_support(backend, simple_docs):
    """End-to-end test: allow_partial_support parameter behavior."""
    response = "Paris is the capital of France and located in northern France."

    ctx = (
        ChatContext()
        .add(Message("user", "Tell me about Paris"))
        .add(Message("assistant", response))
    )

    # Test with strict mode (all spans must be fully supported)
    req_strict = GroundednessRequirement(
        documents=simple_docs, allow_partial_support=False
    )
    result_strict = await req_strict.validate(backend, ctx)

    # Test with lenient mode (partial support is acceptable)
    req_lenient = GroundednessRequirement(
        documents=simple_docs, allow_partial_support=True
    )
    result_lenient = await req_lenient.validate(backend, ctx)

    # Both should return valid results
    assert isinstance(result_strict.as_bool(), bool)
    assert isinstance(result_lenient.as_bool(), bool)


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
async def test_groundedness_e2e_short_factual_response(backend, simple_docs):
    """End-to-end test: short factual response."""
    req = GroundednessRequirement(documents=simple_docs, allow_partial_support=False)

    ctx = (
        ChatContext()
        .add(Message("user", "Capital of France?"))
        .add(Message("assistant", "Paris"))
    )

    result = await req.validate(backend, ctx)
    # Short factual response grounded in documents
    assert result.as_bool() is True


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
async def test_groundedness_e2e_complex_response(backend, simple_docs):
    """End-to-end test: complex multi-fact response."""
    req = GroundednessRequirement(documents=simple_docs, allow_partial_support=False)

    ctx = (
        ChatContext()
        .add(Message("user", "Describe the capitals mentioned in the documents"))
        .add(
            Message(
                "assistant",
                (
                    "Paris is the capital of France and is located in northern France. "
                    "London is the capital of England and is located on the River Thames. "
                    "Both are major European cities."
                ),
            )
        )
    )

    result = await req.validate(backend, ctx)
    # Most claims are grounded, but "Both are major European cities" may not be explicitly supported
    assert isinstance(result.as_bool(), bool)


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
async def test_groundedness_e2e_string_documents(backend):
    """End-to-end test: documents provided as strings instead of Document objects."""
    doc_strings = [
        "The Eiffel Tower is located in Paris, France.",
        "The Tower of London is located in London, England.",
    ]

    req = GroundednessRequirement(documents=doc_strings, allow_partial_support=False)

    ctx = (
        ChatContext()
        .add(Message("user", "Where is the Eiffel Tower?"))
        .add(Message("assistant", "The Eiffel Tower is located in Paris, France."))
    )

    result = await req.validate(backend, ctx)
    # Response is grounded in the provided string documents
    assert result.as_bool() is True


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
async def test_groundedness_e2e_real_world_rag_scenario(backend):
    """End-to-end test: realistic RAG scenario with retrieved documents."""
    # Simulate documents retrieved from a knowledge base
    retrieved_docs = [
        Document(
            doc_id="doc_1",
            title="Python Lists",
            text=(
                "Python lists are ordered, mutable collections that can store multiple items "
                "of different types. Lists are defined using square brackets [] and items are "
                "separated by commas. Lists are zero-indexed, meaning the first item has index 0."
            ),
        ),
        Document(
            doc_id="doc_2",
            title="Python Dictionaries",
            text=(
                "Python dictionaries are unordered collections of key-value pairs. They are "
                "defined using curly braces {} and items are separated by commas. Dictionary "
                "values are accessed using their corresponding keys."
            ),
        ),
    ]

    req = GroundednessRequirement(documents=retrieved_docs, allow_partial_support=False)

    # Simulate a RAG system responding to a user query with retrieved documents
    ctx = (
        ChatContext()
        .add(Message("user", "What is a Python list?"))
        .add(
            Message(
                "assistant",
                (
                    "A Python list is an ordered, mutable collection that can store multiple "
                    "items of different types. Lists are defined using square brackets and "
                    "items are separated by commas. They are zero-indexed, so the first item "
                    "has index 0."
                ),
                documents=retrieved_docs,
            )
        )
    )

    result = await req.validate(backend, ctx)
    # Response is fully grounded in the retrieved documents
    assert result.as_bool() is True


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
async def test_groundedness_e2e_citation_not_needed(backend, simple_docs):
    """End-to-end test: citation not needed when response indicates lack of information."""
    # When assistant explicitly states it cannot answer, no citation is required
    # even though documents are provided
    req = GroundednessRequirement(documents=simple_docs, allow_partial_support=False)

    ctx = (
        ChatContext()
        .add(Message("user", "What is the capital of Germany?"))
        .add(Message("assistant", "I do not have information to answer your question."))
    )

    result = await req.validate(backend, ctx)
    # Requirements checker should return True when citation is not needed
    # (the response does not make factual claims that require grounding)
    assert result.as_bool() is True
    assert result.reason is not None
