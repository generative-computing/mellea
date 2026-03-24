"""Tests for RAG requirements."""
# pytest: huggingface, llm, requires_heavy_ram

import pytest

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.rag import CitationMode, CitationRequirement


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
    """Test citation requirement with non-adapter backend."""
    from unittest.mock import Mock

    # Create a mock backend that doesn't support adapters
    backend = Mock()
    backend.__class__.__name__ = "MockBackend"

    # Create documents
    docs = [Document(doc_id="doc1", text="The sky is blue.")]

    # Create context
    ctx = ChatContext().add(Message("user", "What color is the sky?"))
    ctx = ctx.add(Message("assistant", "The sky is blue.", documents=docs))

    # Create requirement
    req = CitationRequirement(min_citation_coverage=0.8)

    # Validate
    result = await req.validate(backend, ctx)

    # Should fail with clear error about adapter requirement
    assert not result.as_bool()
    assert result.reason is not None
    assert "adapter" in result.reason.lower()


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

    # Empty response should pass (considered to have adequate coverage)
    assert result.as_bool()
    assert result.score == 1.0
    assert result.reason is not None
    assert "adequate citation coverage" in result.reason.lower()


async def test_citation_requirement_threshold_boundary():
    """Test citation requirement at exact threshold boundary.

    This test mocks the find_citations intrinsic to return a controlled
    result that produces exactly the threshold coverage (80%).
    """
    from unittest.mock import Mock, patch

    backend = Mock(spec=LocalHFBackend)

    # Create documents
    docs = [Document(doc_id="doc1", text="The sky is blue during the day.")]

    # Create a response with 10 characters
    response = "1234567890"

    # Create context
    ctx = ChatContext().add(Message("user", "What color is the sky?"))
    ctx = ctx.add(Message("assistant", response, documents=docs))

    # Mock find_citations to return exactly 8 characters cited (80% of 10)
    mock_citations = [
        {
            "response_begin": 0,
            "response_end": 8,  # 8 characters cited
            "response_text": "12345678",
            "citation_doc_id": "doc1",
            "citation_text": "The sky is blue",
        }
    ]

    with patch(
        "mellea.stdlib.components.intrinsic.rag.find_citations",
        return_value=mock_citations,
    ):
        # Test at exact threshold (0.8) with CHARACTERS mode
        req = CitationRequirement(
            min_citation_coverage=0.8, mode=CitationMode.CHARACTERS
        )
        result = await req.validate(backend, ctx)

        # At exact threshold, should pass (>= comparison)
        assert result.as_bool()
        assert result.score == 0.8

        # Test just below threshold (0.81) with CHARACTERS mode
        req_above = CitationRequirement(
            min_citation_coverage=0.81, mode=CitationMode.CHARACTERS
        )
        result_above = await req_above.validate(backend, ctx)

        # Just below threshold, should fail
        assert not result_above.as_bool()
        assert result_above.score == 0.8


async def test_citation_requirement_claims_mode():
    """Test citation requirement with CLAIMS mode."""
    from unittest.mock import Mock, patch

    backend = Mock(spec=LocalHFBackend)

    # Create documents
    docs = [
        Document(doc_id="doc1", text="The sky is blue."),
        Document(doc_id="doc2", text="Grass is green."),
    ]

    # Create a response with 3 sentences
    response = "The sky is blue. Grass is green. Water is wet."

    # Create context
    ctx = ChatContext().add(Message("user", "Tell me some facts."))
    ctx = ctx.add(Message("assistant", response, documents=docs))

    # Mock find_citations to return 2 citations (2 out of 3 claims = 66.7%)
    mock_citations = [
        {
            "response_begin": 0,
            "response_end": 16,
            "response_text": "The sky is blue",
            "citation_doc_id": "doc1",
            "citation_text": "The sky is blue.",
        },
        {
            "response_begin": 18,
            "response_end": 34,
            "response_text": "Grass is green",
            "citation_doc_id": "doc2",
            "citation_text": "Grass is green.",
        },
    ]

    with patch(
        "mellea.stdlib.components.intrinsic.rag.find_citations",
        return_value=mock_citations,
    ):
        # Test with CLAIMS mode (default) - 2 citations out of 3 sentences = 66.7%
        req = CitationRequirement(min_citation_coverage=0.6)
        result = await req.validate(backend, ctx)

        # Should pass (66.7% >= 60%)
        assert result.as_bool()
        assert result.score is not None
        assert abs(result.score - 0.667) < 0.01  # Allow small floating point error
        assert result.reason is not None
        assert "claims" in result.reason

        # Test with higher threshold that should fail
        req_high = CitationRequirement(min_citation_coverage=0.7)
        result_high = await req_high.validate(backend, ctx)

        # Should fail (66.7% < 70%)
        assert not result_high.as_bool()
        assert result_high.score is not None
        assert abs(result_high.score - 0.667) < 0.01


async def test_citation_requirement_characters_vs_claims():
    """Test that CHARACTERS and CLAIMS modes produce different results."""
    from unittest.mock import Mock, patch

    backend = Mock(spec=LocalHFBackend)

    # Create documents
    docs = [Document(doc_id="doc1", text="Short fact.")]

    # Create a response: 1 short sentence with citation, 1 long sentence without
    response = "Short. This is a much longer sentence without any citation support."

    # Create context
    ctx = ChatContext().add(Message("user", "Tell me something."))
    ctx = ctx.add(Message("assistant", response, documents=docs))

    # Mock find_citations to return 1 citation for the short sentence
    mock_citations = [
        {
            "response_begin": 0,
            "response_end": 6,  # "Short." = 6 characters
            "response_text": "Short",
            "citation_doc_id": "doc1",
            "citation_text": "Short fact.",
        }
    ]

    with patch(
        "mellea.stdlib.components.intrinsic.rag.find_citations",
        return_value=mock_citations,
    ):
        # CLAIMS mode: 1 citation out of 2 sentences = 50%
        req_claims = CitationRequirement(
            min_citation_coverage=0.5, mode=CitationMode.CLAIMS
        )
        result_claims = await req_claims.validate(backend, ctx)

        # CHARACTERS mode: 6 characters out of 67 total = ~9%
        req_chars = CitationRequirement(
            min_citation_coverage=0.5, mode=CitationMode.CHARACTERS
        )
        result_chars = await req_chars.validate(backend, ctx)

        # CLAIMS mode should pass (50% >= 50%)
        assert result_claims.as_bool()
        assert result_claims.score == 0.5

        # CHARACTERS mode should fail (~9% < 50%)
        assert not result_chars.as_bool()
        assert result_chars.score is not None
        assert result_chars.score < 0.1
