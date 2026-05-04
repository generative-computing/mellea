"""Tests for GroundednessRequirement."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.backends.huggingface import LocalHFBackend
from mellea.core.base import ModelOutputThunk
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.rag import GroundednessRequirement
from test.predicates import require_gpu


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
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
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
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
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
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
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
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
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
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
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
@pytest.mark.e2e
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
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


@pytest.mark.asyncio
async def test_identify_citation_necessity_with_cited_spans():
    """Test identifying citation necessity with both cited and uncited spans."""
    req = GroundednessRequirement()

    response = "The sky is blue. Cats are mammals."
    citations = [
        {"response_begin": 0, "response_end": 14, "response_text": "The sky is blue"}
    ]

    # Create a mock backend
    mock_backend = AsyncMock()

    # Mock the generate_from_context to return a necessity judgment
    # Format: [{"span_id": 0, "needs_citation": "yes"}, {"span_id": 1, "needs_citation": "no"}]
    mock_output = '[{"span_id": 0, "needs_citation": "yes"}, {"span_id": 1, "needs_citation": "no"}]'
    mock_thunk = MagicMock(spec=ModelOutputThunk)
    mock_thunk.avalue = AsyncMock()
    mock_thunk.value = mock_output

    mock_backend.generate_from_context = AsyncMock(
        return_value=(mock_thunk, ChatContext())
    )

    context = ChatContext().add(
        Message("user", "What color is the sky and what are cats?")
    )

    # Call the function
    span_necessity = await req._identify_citation_necessity(
        response, citations, mock_backend, context
    )

    # Verify that the backend was called
    assert mock_backend.generate_from_context.called

    # Verify the result is a dictionary
    assert isinstance(span_necessity, dict)
    # Should have at least one mapping
    assert len(span_necessity) > 0


@pytest.mark.asyncio
async def test_identify_citation_necessity_empty_response():
    """Test identifying citation necessity with empty response."""
    req = GroundednessRequirement()

    response = ""
    citations = []

    mock_backend = AsyncMock()
    context = ChatContext().add(Message("user", "Test question"))

    span_necessity = await req._identify_citation_necessity(
        response, citations, mock_backend, context
    )

    # Should return empty dict for empty response
    assert span_necessity == {}
    # Backend should not be called for empty response
    assert not mock_backend.generate_from_context.called


@pytest.mark.asyncio
async def test_identify_citation_necessity_backend_failure():
    """Test handling of backend failure in citation necessity assessment."""
    req = GroundednessRequirement()

    response = "The sky is blue."
    citations = [
        {"response_begin": 0, "response_end": 14, "response_text": "The sky is blue"}
    ]

    mock_backend = AsyncMock()
    mock_backend.generate_from_context = AsyncMock(
        side_effect=ValueError("Backend error")
    )

    context = ChatContext().add(Message("user", "What color is the sky?"))

    # Should raise ValueError when backend fails
    with pytest.raises(ValueError, match="LLM judgment failed"):
        await req._identify_citation_necessity(
            response, citations, mock_backend, context
        )


@pytest.mark.asyncio
async def test_identify_citation_necessity_none_output():
    """Test handling of None output from backend."""
    req = GroundednessRequirement()

    response = "The sky is blue."
    citations = [
        {"response_begin": 0, "response_end": 14, "response_text": "The sky is blue"}
    ]

    mock_backend = AsyncMock()
    mock_thunk = MagicMock(spec=ModelOutputThunk)
    mock_thunk.avalue = AsyncMock()
    mock_thunk.value = None

    mock_backend.generate_from_context = AsyncMock(
        return_value=(mock_thunk, ChatContext())
    )

    context = ChatContext().add(Message("user", "What color is the sky?"))

    # Should raise ValueError when output is None
    with pytest.raises(ValueError, match="LLM judgment returned None"):
        await req._identify_citation_necessity(
            response, citations, mock_backend, context
        )


@pytest.mark.asyncio
async def test_assess_citation_support_overlap_edge_case(sample_docs):
    """Test citation support assessment with edge cases in span-citation overlap.

    This tests the scenario where a span is marked as not having citations,
    but citations might partially overlap due to boundary/whitespace issues.
    """
    req = GroundednessRequirement()

    # Response with potential whitespace boundary issues
    response = "Fact one. Fact two."
    # Citation covers "Fact one"
    citations = [
        {
            "response_begin": 0,
            "response_end": 9,
            "citation_text": "Fact one",
            "citation_doc_id": "0",
        }
    ]

    # Span that needs citations but isn't fully covered
    span_necessity = {
        (11, 20): True  # "Fact two" needs citation but isn't covered by citations
    }

    mock_backend = AsyncMock()
    mock_output = '[{"support_level": "NOT_SUPPORTED"}]'
    mock_thunk = MagicMock(spec=ModelOutputThunk)
    mock_thunk.avalue = AsyncMock()
    mock_thunk.value = mock_output

    mock_backend.generate_from_context = AsyncMock(
        return_value=(mock_thunk, ChatContext())
    )

    context = ChatContext().add(Message("user", "Test question"))

    span_support = await req._assess_citation_support(
        response, citations, span_necessity, mock_backend, context, sample_docs
    )

    # Should attempt to assess support even though span isn't covered by citations
    assert (11, 20) in span_support


@pytest.mark.asyncio
async def test_identify_citation_necessity_prompt_as_action():
    """Test that the necessity prompt is passed as the action, not as a context message."""
    req = GroundednessRequirement()

    response = "The sky is blue."
    citations = [
        {"response_begin": 0, "response_end": 14, "response_text": "The sky is blue"}
    ]

    mock_backend = AsyncMock()
    mock_output = '[{"span_id": 0, "needs_citation": "yes"}]'
    mock_thunk = MagicMock(spec=ModelOutputThunk)
    mock_thunk.avalue = AsyncMock()
    mock_thunk.value = mock_output

    mock_backend.generate_from_context = AsyncMock(
        return_value=(mock_thunk, ChatContext())
    )

    context = ChatContext().add(Message("user", "Original context message"))

    await req._identify_citation_necessity(response, citations, mock_backend, context)

    # Verify generate_from_context was called with correct parameters
    call_args = mock_backend.generate_from_context.call_args
    assert call_args is not None

    # The action (first argument) should be a CBlock with the necessity prompt
    action = call_args[0][0]
    assert action is not None
    # The action should contain the prompt (as a CBlock)
    assert hasattr(action, "content") or hasattr(action, "__str__")

    # The context (second argument) should be the original context
    called_context = call_args[0][1]
    messages = called_context.as_list()
    assert len(messages) == 1  # Only the original message
    assert messages[0].role == "user"
    assert messages[0].content == "Original context message"


def test_build_batch_support_prompt(sample_docs):
    """Test building the batch support prompt for multiple spans."""
    req = GroundednessRequirement()

    response = "The sky is blue. Grass is green."
    spans_to_assess = [
        {
            "text": "The sky is blue",
            "citations": [
                {
                    "citation_text": "The sky appears blue due to Rayleigh scattering",
                    "citation_doc_id": "0",
                }
            ],
        },
        {
            "text": "Grass is green",
            "citations": [
                {
                    "citation_text": "Grass contains chlorophyll which is green",
                    "citation_doc_id": "1",
                }
            ],
        },
    ]

    prompt = req._build_batch_support_prompt(response, spans_to_assess, sample_docs)

    # Verify prompt structure
    assert "JSON array" in prompt or "json" in prompt.lower()
    assert "span_id" in prompt
    assert "support_level" in prompt
    assert "FULLY_SUPPORTED" in prompt or "fully" in prompt.lower()
    assert "PARTIALLY_SUPPORTED" in prompt or "partially" in prompt.lower()
    assert "NOT_SUPPORTED" in prompt or "not" in prompt.lower()
    assert "The sky is blue" in prompt
    assert "Grass is green" in prompt


def test_parse_batch_support_output():
    """Test parsing batch support output."""
    req = GroundednessRequirement()

    # Simulate LLM output: two spans with different support levels
    output_text = """
    [
        {"span_id": 0, "support_level": "FULLY_SUPPORTED"},
        {"span_id": 1, "support_level": "PARTIALLY_SUPPORTED"}
    ]
    """

    result = req._parse_batch_support_output(output_text, 2)

    assert len(result) == 2
    assert result[0] == "FULLY_SUPPORTED"
    assert result[1] == "PARTIALLY_SUPPORTED"


def test_parse_batch_support_output_with_recovery():
    """Test parsing batch support output with malformed JSON recovery."""
    req = GroundednessRequirement()

    # Simulate malformed LLM output that needs recovery
    output_text = """
    Some preamble text
    [
        {"span_id": 0, "support_level": "FULLY_SUPPORTED"},
        {"span_id": 1, "support_level": "NOT_SUPPORTED"
    ]
    """

    result = req._parse_batch_support_output(output_text, 2)

    # Should recover and parse both spans
    assert 0 in result
    assert 1 in result
    assert result[0] == "FULLY_SUPPORTED"
    assert result[1] == "NOT_SUPPORTED"


def test_parse_batch_support_output_missing_spans():
    """Test parsing batch support output when some spans are missing."""
    req = GroundednessRequirement()

    # Output only has one span, but we expect two
    output_text = '[{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]'

    result = req._parse_batch_support_output(output_text, 2)

    # Should have default NOT_SUPPORTED for missing span
    assert len(result) == 2
    assert result[0] == "FULLY_SUPPORTED"
    assert result[1] == "NOT_SUPPORTED"  # Default


def test_parse_batch_support_output_invalid_json():
    """Test parsing batch support output with invalid JSON."""
    req = GroundednessRequirement()

    # Invalid JSON
    output_text = "This is not JSON at all"

    result = req._parse_batch_support_output(output_text, 2)

    # Should conservatively default all to NOT_SUPPORTED
    assert len(result) == 2
    assert result[0] == "NOT_SUPPORTED"
    assert result[1] == "NOT_SUPPORTED"
