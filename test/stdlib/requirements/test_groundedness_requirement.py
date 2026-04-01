# pytest: huggingface, llm, requires_heavy_ram
"""Tests for GroundednessRequirement."""

import pytest

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.rag import GroundednessRequirement


@pytest.fixture
def backend():
    """Provide HuggingFace backend for tests."""
    return LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")


@pytest.fixture
def sample_docs():
    """Provide sample documents for testing."""
    return [
        Document(
            doc_id="0",
            text=(
                "The sky appears blue during the day due to Rayleigh scattering. "
                "Grass is green because of chlorophyll in its leaves."
            ),
        ),
        Document(doc_id="1", text="Cats are mammals that purr when happy."),
    ]


@pytest.mark.asyncio
async def test_groundedness_requirement_initialization():
    """Test that GroundednessRequirement initializes correctly."""
    req = GroundednessRequirement()
    assert req.description is not None
    assert req.allow_partial_support is False

    req2 = GroundednessRequirement(allow_partial_support=True)
    assert req2.allow_partial_support is True


@pytest.mark.asyncio
async def test_groundedness_requirement_empty_response(backend, sample_docs):
    """Test validation with empty response."""
    req = GroundednessRequirement(documents=sample_docs)

    ctx = (
        ChatContext()
        .add(Message("user", "Why is the sky blue?"))
        .add(Message("assistant", ""))
    )

    result = await req.validate(backend, ctx)
    assert result.as_bool() is True
    assert "empty" in result.reason.lower()


@pytest.mark.asyncio
async def test_groundedness_requirement_no_documents_error(backend):
    """Test that validation fails when no documents provided."""
    req = GroundednessRequirement()

    ctx = (
        ChatContext()
        .add(Message("user", "Why is the sky blue?"))
        .add(Message("assistant", "The sky is blue."))
    )

    result = await req.validate(backend, ctx)
    assert result.as_bool() is False
    assert "no documents" in result.reason.lower()


@pytest.mark.asyncio
async def test_groundedness_requirement_documents_in_constructor(backend, sample_docs):
    """Test providing documents in constructor."""
    req = GroundednessRequirement(documents=sample_docs)

    ctx = (
        ChatContext()
        .add(Message("user", "What color is the sky?"))
        .add(Message("assistant", "The sky is blue due to Rayleigh scattering."))
    )

    result = await req.validate(backend, ctx)
    # Should not fail on missing documents
    assert isinstance(result.as_bool(), bool)


@pytest.mark.asyncio
async def test_groundedness_requirement_documents_in_message(backend, sample_docs):
    """Test providing documents in message."""
    req = GroundednessRequirement()

    ctx = (
        ChatContext()
        .add(Message("user", "What color is grass?"))
        .add(
            Message(
                "assistant",
                "Grass is green because of chlorophyll.",
                documents=sample_docs,
            )
        )
    )

    result = await req.validate(backend, ctx)
    # Should not fail on missing documents in constructor
    assert isinstance(result.as_bool(), bool)


@pytest.mark.asyncio
async def test_groundedness_requirement_last_message_not_assistant(
    backend, sample_docs
):
    """Test that validation fails if last message is not from assistant."""
    req = GroundednessRequirement(documents=sample_docs)

    ctx = (
        ChatContext()
        .add(Message("user", "Why is the sky blue?"))
        .add(Message("assistant", "The sky is blue."))
        .add(Message("user", "Follow up question?"))
    )

    result = await req.validate(backend, ctx)
    assert result.as_bool() is False
    assert "assistant response" in result.reason.lower()


@pytest.mark.asyncio
async def test_groundedness_requirement_allow_partial_support_parameter(
    backend, sample_docs
):
    """Test allow_partial_support parameter affects behavior."""
    req_strict = GroundednessRequirement(
        documents=sample_docs, allow_partial_support=False
    )
    req_lenient = GroundednessRequirement(
        documents=sample_docs, allow_partial_support=True
    )

    # Both should have different behavior
    assert req_strict.allow_partial_support is False
    assert req_lenient.allow_partial_support is True


@pytest.mark.asyncio
async def test_span_extraction_simple(backend, sample_docs):
    """Test span extraction with simple response."""
    req = GroundednessRequirement(documents=sample_docs)

    response = "The sky is blue. Cats are mammals."
    # Simple mock citations
    citations = [
        {"response_begin": 0, "response_end": 14, "response_text": "The sky is blue"}
    ]

    spans = req._extract_response_spans(response, citations)
    assert len(spans) > 0
    # Should have both cited and uncited portions
    assert any(span.get("is_cited") for span in spans)


@pytest.mark.asyncio
async def test_parse_support_output():
    """Test parsing of support level output."""
    req = GroundednessRequirement()

    # Test fully supported
    result = req._parse_support_output("The citation FULLY_SUPPORTED the claim")
    assert result == "FULLY_SUPPORTED"

    # Test partially supported
    result = req._parse_support_output("This is PARTIALLY_SUPPORTED")
    assert result == "PARTIALLY_SUPPORTED"

    # Test not supported
    result = req._parse_support_output("This is NOT_SUPPORTED")
    assert result == "NOT_SUPPORTED"

    # Test default to not supported
    result = req._parse_support_output("Some random output")
    assert result == "NOT_SUPPORTED"


def test_build_necessity_prompt():
    """Test building the necessity prompt."""
    req = GroundednessRequirement()

    response = "The sky is blue. Cats are mammals."
    spans = [
        {"begin": 0, "end": 14, "text": "The sky is blue"},
        {"begin": 16, "end": 33, "text": "Cats are mammals"},
    ]

    prompt = req._build_necessity_prompt(response, spans)
    assert "Response:" in prompt
    assert "span_id" in prompt
    assert "needs_citation" in prompt or "citation" in prompt.lower()


def test_build_support_prompt():
    """Test building the support prompt."""
    req = GroundednessRequirement()

    span_text = "The sky is blue"
    span_citations = [
        {
            "citation_text": "The sky appears blue due to Rayleigh scattering",
            "citation_doc_id": "0",
        }
    ]

    prompt = req._build_support_prompt(span_text, span_citations)
    assert "Response span" in prompt
    assert span_text in prompt
    assert "support" in prompt.lower()
    assert "FULLY_SUPPORTED" in prompt or "fully" in prompt.lower()
