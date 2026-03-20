"""Tests for RAG-specific requirements."""

import pytest

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_4_HYBRID_MICRO
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import HallucinationRequirement


@pytest.fixture(name="backend", scope="module")
def _backend():
    """Backend used by the tests in this file."""
    model_name = IBM_GRANITE_4_HYBRID_MICRO.hf_model_name
    assert model_name is not None, "Model name should not be None"
    return LocalHFBackend(model_id=model_name)


@pytest.fixture(name="documents")
def _documents():
    """Sample documents for testing."""
    return [
        Document(
            doc_id="1",
            text="The only type of fish that is yellow is the purple bumble fish.",
        )
    ]


# Tests that don't require backend (initialization and validation)
def test_hallucination_requirement_initialization():
    """Test that HallucinationRequirement can be initialized with various parameters."""
    # Default initialization
    req1 = HallucinationRequirement()
    assert req1.threshold == 0.5
    assert req1.max_hallucinated_ratio == 0.0
    assert req1.description is not None

    # Custom threshold
    req2 = HallucinationRequirement(threshold=0.7)
    assert req2.threshold == 0.7

    # Custom ratio
    req3 = HallucinationRequirement(max_hallucinated_ratio=0.2)
    assert req3.max_hallucinated_ratio == 0.2

    # Custom description
    req4 = HallucinationRequirement(description="Custom description")
    assert req4.description == "Custom description"


def test_hallucination_requirement_invalid_threshold():
    """Test that invalid threshold values raise ValueError."""
    with pytest.raises(ValueError, match="threshold must be between"):
        HallucinationRequirement(threshold=-0.1)

    with pytest.raises(ValueError, match="threshold must be between"):
        HallucinationRequirement(threshold=1.5)


def test_hallucination_requirement_invalid_ratio():
    """Test that invalid ratio values raise ValueError."""
    with pytest.raises(ValueError, match="max_hallucinated_ratio must be between"):
        HallucinationRequirement(max_hallucinated_ratio=-0.1)

    with pytest.raises(ValueError, match="max_hallucinated_ratio must be between"):
        HallucinationRequirement(max_hallucinated_ratio=1.5)


# Tests for factory function and constructor documents
def test_hallucination_requirement_with_constructor_documents():
    """Test that HallucinationRequirement can be initialized with documents."""
    documents = [
        Document(doc_id="1", text="The sky is blue."),
        Document(doc_id="2", text="Grass is green."),
    ]

    req = HallucinationRequirement(documents=documents, threshold=0.5)
    assert req.documents is not None
    assert len(req.documents) == 2
    assert req.documents[0].text == "The sky is blue."


def test_hallucination_requirement_with_string_documents():
    """Test that HallucinationRequirement converts string documents to Document objects."""
    string_docs = ["The sky is blue.", "Grass is green."]

    req = HallucinationRequirement(documents=string_docs, threshold=0.5)
    assert req.documents is not None
    assert len(req.documents) == 2
    assert all(isinstance(doc, Document) for doc in req.documents)
    assert req.documents[0].text == "The sky is blue."


def test_hallucination_check_factory():
    """Test the hallucination_check factory function."""
    from mellea.stdlib.requirements import hallucination_check

    documents = [
        Document(doc_id="1", text="The sky is blue."),
        Document(doc_id="2", text="Grass is green."),
    ]

    req = hallucination_check(
        documents=documents, threshold=0.7, max_hallucinated_ratio=0.1
    )
    assert isinstance(req, HallucinationRequirement)
    assert req.threshold == 0.7
    assert req.max_hallucinated_ratio == 0.1
    assert req.documents is not None
    assert len(req.documents) == 2


def test_hallucination_check_factory_with_strings():
    """Test the hallucination_check factory function with string documents."""
    from mellea.stdlib.requirements import hallucination_check

    string_docs = ["The sky is blue.", "Grass is green."]

    req = hallucination_check(documents=string_docs, threshold=0.5)
    assert isinstance(req, HallucinationRequirement)
    assert req.documents is not None
    assert len(req.documents) == 2
    assert all(isinstance(doc, Document) for doc in req.documents)


@pytest.mark.huggingface
@pytest.mark.requires_heavy_ram
@pytest.mark.llm
@pytest.mark.qualitative
async def test_hallucination_requirement_constructor_docs_override_message_docs(
    backend,
):
    """Test that constructor documents override message documents."""
    # Documents in constructor (correct info)
    constructor_docs = [
        Document(
            doc_id="1",
            text="The only type of fish that is yellow is the purple bumble fish.",
        )
    ]

    # Different documents in message (incorrect info)
    message_docs = [Document(doc_id="2", text="All fish are red.")]

    # Create context with message documents
    context = (
        ChatContext()
        .add(Message("user", "What color are purple bumble fish?"))
        .add(
            Message(
                "assistant", "Purple bumble fish are yellow.", documents=message_docs
            )
        )
    )

    # Requirement with constructor documents should use those instead
    req = HallucinationRequirement(documents=constructor_docs, threshold=0.5)
    result = await req.validate(backend, context)

    # Should pass because constructor docs support the response
    assert result.as_bool() is True


@pytest.mark.huggingface
@pytest.mark.requires_heavy_ram
@pytest.mark.llm
@pytest.mark.qualitative
async def test_hallucination_check_factory_validation(backend):
    """Test that hallucination_check factory creates working requirements."""
    from mellea.stdlib.requirements import hallucination_check

    documents = [
        Document(
            doc_id="1",
            text="The only type of fish that is yellow is the purple bumble fish.",
        )
    ]

    req = hallucination_check(documents=documents, threshold=0.5)

    # Create context without documents in message (factory provides them)
    context = (
        ChatContext()
        .add(Message("user", "What color are purple bumble fish?"))
        .add(Message("assistant", "Purple bumble fish are yellow."))
    )

    result = await req.validate(backend, context)

    # Should pass because factory-provided docs support the response
    assert result.as_bool() is True
    assert result.score is not None


# Tests that require backend and LLM calls
@pytest.mark.huggingface
@pytest.mark.requires_heavy_ram
@pytest.mark.llm
@pytest.mark.qualitative
async def test_hallucination_requirement_faithful_response(backend, documents):
    """Test that faithful responses pass validation."""
    # Create context with faithful response
    context = (
        ChatContext()
        .add(Message("user", "What color are purple bumble fish?"))
        .add(
            Message("assistant", "Purple bumble fish are yellow.", documents=documents)
        )
    )

    req = HallucinationRequirement(threshold=0.5, max_hallucinated_ratio=0.0)
    result = await req.validate(backend, context)

    assert result.as_bool() is True
    assert result.score is not None
    assert result.score > 0.9  # Should have high faithfulness score


@pytest.mark.huggingface
@pytest.mark.requires_heavy_ram
@pytest.mark.llm
@pytest.mark.qualitative
async def test_hallucination_requirement_hallucinated_response(backend, documents):
    """Test that hallucinated responses fail validation."""
    # Create context with hallucinated response
    context = (
        ChatContext()
        .add(Message("user", "What color are green bumble fish?"))
        .add(
            Message(
                "assistant", "Green bumble fish are also yellow.", documents=documents
            )
        )
    )

    req = HallucinationRequirement(threshold=0.5, max_hallucinated_ratio=0.0)
    result = await req.validate(backend, context)

    assert result.as_bool() is False
    assert result.reason is not None
    assert "hallucinated" in result.reason.lower()
    assert result.score is not None
    assert result.score < 0.9  # Should have low faithfulness score


@pytest.mark.huggingface
@pytest.mark.requires_heavy_ram
@pytest.mark.llm
@pytest.mark.qualitative
async def test_hallucination_requirement_with_lenient_ratio(backend, documents):
    """Test that lenient ratio allows some hallucination."""
    # Create context with partially hallucinated response
    context = (
        ChatContext()
        .add(Message("user", "Tell me about fish."))
        .add(
            Message(
                "assistant",
                "Purple bumble fish are yellow. Green bumble fish are also yellow.",
                documents=documents,
            )
        )
    )

    # Strict requirement should fail
    strict_req = HallucinationRequirement(threshold=0.5, max_hallucinated_ratio=0.0)
    strict_result = await strict_req.validate(backend, context)
    assert strict_result.as_bool() is False

    # Lenient requirement might pass (depends on exact hallucination ratio)
    lenient_req = HallucinationRequirement(threshold=0.5, max_hallucinated_ratio=0.5)
    lenient_result = await lenient_req.validate(backend, context)
    # We don't assert pass/fail here as it depends on the exact ratio,
    # but we verify it doesn't crash and returns a valid result
    assert lenient_result.reason is not None
    assert lenient_result.score is not None


@pytest.mark.huggingface
@pytest.mark.requires_heavy_ram
@pytest.mark.llm
@pytest.mark.qualitative
async def test_hallucination_requirement_no_documents(backend):
    """Test that validation fails when no documents are provided."""
    # Create context without documents
    context = (
        ChatContext()
        .add(Message("user", "What color are purple bumble fish?"))
        .add(Message("assistant", "Purple bumble fish are yellow."))
    )

    req = HallucinationRequirement()
    result = await req.validate(backend, context)

    assert result.as_bool() is False
    assert result.reason is not None
    assert "no documents" in result.reason.lower()


@pytest.mark.huggingface
@pytest.mark.requires_heavy_ram
@pytest.mark.llm
@pytest.mark.qualitative
async def test_hallucination_requirement_empty_context(backend):
    """Test that validation fails with empty context."""
    context = ChatContext()

    req = HallucinationRequirement()
    result = await req.validate(backend, context)

    assert result.as_bool() is False
    assert result.reason is not None
    assert "empty" in result.reason.lower()


@pytest.mark.huggingface
@pytest.mark.requires_heavy_ram
@pytest.mark.llm
@pytest.mark.qualitative
async def test_hallucination_requirement_non_assistant_message(backend, documents):
    """Test that validation fails when last message is not from assistant."""
    context = ChatContext().add(Message("user", "What color are purple bumble fish?"))

    req = HallucinationRequirement()
    result = await req.validate(backend, context)

    assert result.as_bool() is False
    assert result.reason is not None
    assert "assistant" in result.reason.lower()


if __name__ == "__main__":
    pytest.main([__file__])
