"""Tests for RAG requirements."""
# pytest: huggingface, llm, requires_heavy_ram

import pytest

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.rag import CitationRequirement, citation_check


@pytest.mark.huggingface
@pytest.mark.llm
@pytest.mark.requires_heavy_ram
async def test_citation_requirement_basic():
    """Test basic citation requirement functionality."""
    backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

    # Create documents
    docs = [
        Document(doc_id="doc1", text="The sky is blue during the day."),
        Document(doc_id="doc2", text="Grass is typically green in color."),
    ]

    # Create a response that should have citations
    response = "The sky is blue. Grass is green."

    # Create context with assistant message
    ctx = ChatContext().add(Message("user", "What colors are the sky and grass?"))
    ctx = ctx.add(Message("assistant", response, documents=docs))

    # Create requirement
    req = CitationRequirement(min_citation_coverage=0.5)

    # Validate
    result = await req.validate(backend, ctx)

    # Should pass if citations are found
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert result.reason is not None


@pytest.mark.huggingface
@pytest.mark.llm
@pytest.mark.requires_heavy_ram
async def test_citation_requirement_with_constructor_documents():
    """Test citation requirement with documents in constructor."""
    backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

    # Create documents
    docs = [
        Document(doc_id="doc1", text="The sky is blue during the day."),
        Document(doc_id="doc2", text="Grass is typically green in color."),
    ]

    # Create a response
    response = "The sky is blue. Grass is green."

    # Create context with assistant message (no documents attached)
    ctx = ChatContext().add(Message("user", "What colors are the sky and grass?"))
    ctx = ctx.add(Message("assistant", response))

    # Create requirement with documents in constructor
    req = CitationRequirement(min_citation_coverage=0.5, documents=docs)

    # Validate
    result = await req.validate(backend, ctx)

    # Should use constructor documents
    assert isinstance(result.score, float)
    assert result.reason is not None


@pytest.mark.huggingface
@pytest.mark.llm
@pytest.mark.requires_heavy_ram
async def test_citation_check_factory():
    """Test citation_check factory function."""
    backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

    # Create documents
    docs = [Document(doc_id="doc1", text="The sky is blue during the day.")]

    # Create a response
    response = "The sky is blue."

    # Create context
    ctx = ChatContext().add(Message("user", "What color is the sky?"))
    ctx = ctx.add(Message("assistant", response))

    # Use factory function
    req = citation_check(docs, min_citation_coverage=0.5)

    # Validate
    result = await req.validate(backend, ctx)

    # Should work the same as CitationRequirement
    assert isinstance(result.score, float)
    assert result.reason is not None


async def test_citation_requirement_empty_context():
    """Test citation requirement with empty context."""
    # Create a mock backend - we don't need a real one for this test
    # since validation fails before backend is used
    from unittest.mock import Mock

    backend = Mock(spec=LocalHFBackend)

    # Create empty context
    ctx = ChatContext()

    # Create requirement
    req = CitationRequirement(min_citation_coverage=0.8)

    # Validate
    result = await req.validate(backend, ctx)

    # Should fail with clear error
    assert not result.as_bool()
    assert result.reason is not None
    assert "empty" in result.reason.lower()


async def test_citation_requirement_wrong_message_role():
    """Test citation requirement with non-assistant last message."""
    # Create a mock backend - we don't need a real one for this test
    from unittest.mock import Mock

    backend = Mock(spec=LocalHFBackend)

    # Create context ending with user message
    ctx = ChatContext().add(Message("user", "What color is the sky?"))

    # Create requirement
    req = CitationRequirement(min_citation_coverage=0.8)

    # Validate
    result = await req.validate(backend, ctx)

    # Should fail with clear error
    assert not result.as_bool()
    assert result.reason is not None
    assert "assistant" in result.reason.lower()


async def test_citation_requirement_no_documents():
    """Test citation requirement with no documents provided."""
    # Create a mock backend - we don't need a real one for this test
    from unittest.mock import Mock

    backend = Mock(spec=LocalHFBackend)

    # Create context without documents
    ctx = ChatContext().add(Message("user", "What color is the sky?"))
    ctx = ctx.add(Message("assistant", "The sky is blue."))

    # Create requirement without documents
    req = CitationRequirement(min_citation_coverage=0.8)

    # Validate
    result = await req.validate(backend, ctx)

    # Should fail with clear error about missing documents
    assert not result.as_bool()
    assert result.reason is not None
    assert "documents" in result.reason.lower()


async def test_citation_requirement_wrong_backend():
    """Test citation requirement with non-HuggingFace backend."""
    try:
        from mellea.backends.ollama import OllamaBackend  # type: ignore
    except ImportError:
        pytest.skip("Ollama backend not available")

    backend = OllamaBackend(model_id="llama3.2")  # type: ignore

    # Create documents
    docs = [Document(doc_id="doc1", text="The sky is blue.")]

    # Create context
    ctx = ChatContext().add(Message("user", "What color is the sky?"))
    ctx = ctx.add(Message("assistant", "The sky is blue.", documents=docs))

    # Create requirement
    req = CitationRequirement(min_citation_coverage=0.8)

    # Validate
    result = await req.validate(backend, ctx)

    # Should fail with clear error about backend requirement
    assert not result.as_bool()
    assert result.reason is not None
    assert "LocalHFBackend" in result.reason or "HuggingFace" in result.reason


def test_citation_requirement_invalid_coverage():
    """Test citation requirement with invalid coverage values."""
    # Test coverage > 1.0
    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        CitationRequirement(min_citation_coverage=1.5)

    # Test coverage < 0.0
    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        CitationRequirement(min_citation_coverage=-0.5)


def test_citation_requirement_string_documents():
    """Test citation requirement with string documents."""
    # Should convert strings to Document objects
    req = CitationRequirement(
        min_citation_coverage=0.8, documents=["The sky is blue.", "Grass is green."]
    )

    # Check documents were converted
    assert req.documents is not None
    assert len(req.documents) == 2
    assert all(isinstance(doc, Document) for doc in req.documents)


def test_citation_requirement_custom_description():
    """Test citation requirement with custom description."""
    custom_desc = "Custom citation requirement description"
    req = CitationRequirement(min_citation_coverage=0.8, description=custom_desc)

    assert req.description == custom_desc


def test_citation_requirement_default_description():
    """Test citation requirement generates default description."""
    req = CitationRequirement(min_citation_coverage=0.75)

    assert req.description is not None
    assert "75%" in req.description or "0.75" in req.description


@pytest.mark.huggingface
@pytest.mark.llm
@pytest.mark.requires_heavy_ram
async def test_citation_requirement_empty_response():
    """Test citation requirement with empty response."""
    backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

    # Create documents
    docs = [Document(doc_id="doc1", text="The sky is blue.")]

    # Create context with empty response
    ctx = ChatContext().add(Message("user", "What color is the sky?"))
    ctx = ctx.add(Message("assistant", "", documents=docs))

    # Create requirement
    req = CitationRequirement(min_citation_coverage=0.8)

    # Validate
    result = await req.validate(backend, ctx)

    # Empty response should pass (100% coverage of nothing)
    assert result.as_bool()
    assert result.score == 1.0


@pytest.mark.huggingface
@pytest.mark.llm
@pytest.mark.requires_heavy_ram
async def test_citation_requirement_threshold_boundary():
    """Test citation requirement at exact threshold boundary."""
    backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

    # Create documents
    docs = [Document(doc_id="doc1", text="The sky is blue during the day.")]

    # Create a response
    response = "The sky is blue."

    # Create context
    ctx = ChatContext().add(Message("user", "What color is the sky?"))
    ctx = ctx.add(Message("assistant", response, documents=docs))

    # Create requirement with specific threshold
    req = CitationRequirement(min_citation_coverage=0.8)

    # Validate
    result = await req.validate(backend, ctx)

    # Check that score is calculated
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0

    # Result should match threshold comparison
    if result.score >= 0.8:
        assert result.as_bool()
    else:
        assert not result.as_bool()
