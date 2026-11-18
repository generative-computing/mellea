# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GroundednessRequirement."""

import itertools
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.backends.adapters import AdapterMixin
from mellea.backends.model_options import ModelOption
from mellea.core import Requirement, ValidationResult
from mellea.core.base import ModelOutputThunk
from mellea.stdlib.components import Document, Instruction, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.rag import (
    GroundednessRequirement,
    _normalise_needs_citation,
)
from mellea.stdlib.sampling import ModelFriendlyRepairStrategy, RepairTemplateStrategy
from test.conftest import hf_skip
from test.predicates import require_gpu


@pytest.fixture
def backend():
    """Provide HuggingFace backend for tests."""
    try:
        from mellea.backends.huggingface import LocalHFBackend
    except ImportError as exc:
        pytest.skip(str(exc))

    with hf_skip():
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
    assert result.reason is None
    assert "no documents" in str(result.error).lower()


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
    assert result.reason is None
    assert "assistant response" in str(result.error).lower()


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
async def test_span_extraction_simple(sample_docs):
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
async def test_assess_citation_support_skips_llm_when_no_overlap():
    """Test that citation support assessment early-exits without LLM when span has no citation overlap.

    When a span requires citations but no citations overlap with it, the method
    should immediately assign NOT_SUPPORTED without calling the backend.
    """
    req = GroundednessRequirement()

    # Response with two separate sentences
    response = "Fact one. Fact two."
    # Citation covers "Fact one" (0-9)
    citations = [
        {
            "response_begin": 0,
            "response_end": 9,
            "citation_text": "Fact one",
            "citation_doc_id": "0",
        }
    ]

    # Span "Fact two" (11-20) needs citation but has no citation overlap
    span_necessity = {(11, 20): True}

    # Backend should not be called since there's no overlap
    mock_backend = AsyncMock()
    context = ChatContext().add(Message("user", "Test question"))

    span_support = await req._assess_citation_support(
        response, citations, span_necessity, mock_backend, context, []
    )

    # Span should be marked NOT_SUPPORTED without any backend call
    assert span_support[(11, 20)] == "NOT_SUPPORTED"
    mock_backend.generate_from_context.assert_not_called()


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

    # Input spans are keyed "evidence", not "citations" (issue #1316).
    assert '"evidence": [' in prompt
    assert '"citations": [' not in prompt
    assert "Evidence 0" in prompt


def test_build_batch_support_prompt_escapes_special_characters(sample_docs):
    """Text with quotes/newlines must not break the prompt's JSON structure."""
    req = GroundednessRequirement()

    response = 'He said "hello" to me.'
    spans_to_assess = [
        {
            "text": 'He said "hello" to me',
            "citations": [
                {
                    "citation_text": 'The greeting was "hello".\nA second line.',
                    "citation_doc_id": "0",
                }
            ],
        }
    ]

    prompt = req._build_batch_support_prompt(response, spans_to_assess, sample_docs)

    spans_section = prompt[
        prompt.index("Spans to assess:\n") + len("Spans to assess:\n") : prompt.index(
            "\n\nJSON Output:"
        )
    ]
    parsed = json.loads(spans_section)

    assert parsed[0]["span_id"] == 0
    assert parsed[0]["text"] == 'He said "hello" to me'
    assert 'The greeting was "hello".\nA second line.' in parsed[0]["evidence"][0]


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


def test_parse_batch_support_output_normalises_string_span_id():
    """Quoted numeric span IDs must map to the integer result key.

    Without this normalisation, a valid model judgment is stored under `"0"`
    and the caller falls back to NOT_SUPPORTED when looking up integer key `0`.
    """
    req = GroundednessRequirement()

    result = req._parse_batch_support_output(
        '[{"span_id": "0", "support_level": "FULLY_SUPPORTED"}]', 1
    )

    assert result == {0: "FULLY_SUPPORTED"}


def test_parse_batch_support_output_ignores_non_string_support_level():
    """Unexpected support-level types must degrade to unknown."""
    req = GroundednessRequirement()

    result = req._parse_batch_support_output('[{"span_id": 0, "support_level": 5}]', 1)

    assert result == {0: None}


def test_parse_batch_support_output_empty_top_level_support_level_with_valid_nested_evidence():
    """Explicitly empty/unreadable top-level support label must produce None, not fall back to nested evidence."""
    req = GroundednessRequirement()

    output_text = """
    [
        {
            "span_id": 0,
            "support_level": "",
            "evidence": [
                {"support_level": "FULLY_SUPPORTED"}
            ]
        }
    ]
    """
    result = req._parse_batch_support_output(output_text, 1)
    assert result == {0: None}


def test_parse_batch_support_output_non_list_evidence_with_valid_citations():
    """A non-list evidence field alongside valid citations must make the judgment unknown."""
    req = GroundednessRequirement()

    output_text = """
    [
        {
            "span_id": 0,
            "evidence": "malformed",
            "citations": [
                {"support_level": "FULLY_SUPPORTED"}
            ]
        }
    ]
    """
    result = req._parse_batch_support_output(output_text, 1)
    assert result == {0: None}


@pytest.mark.parametrize("nested_key", ["evidence", "citations"])
def test_parse_batch_support_output_ignores_non_string_nested_support_level(
    nested_key: str,
):
    """Unexpected nested support-level types must not crash validation."""
    req = GroundednessRequirement()

    result = req._parse_batch_support_output(
        f'[{{"span_id": 0, "{nested_key}": [{{"support_level": 5}}]}}]', 1
    )

    assert result == {0: None}


@pytest.mark.parametrize(
    "span_id", [True, -1, 1.5, "not-an-index", "2", "²", "1_0", "+1", "３"]
)
def test_parse_batch_support_output_ignores_invalid_span_id(span_id: object):
    """Invalid model-returned span IDs must not become result keys.

    The expected span is then missing from the output, so it is unknown.
    """
    req = GroundednessRequirement()

    result = req._parse_batch_support_output(
        json.dumps([{"span_id": span_id, "support_level": "FULLY_SUPPORTED"}]), 1
    )

    assert result == {0: None}


@pytest.mark.parametrize("span_id", [0.0, "0"])
def test_parse_batch_support_output_accepts_integral_span_id(span_id: object):
    """Integral numeric span IDs must resolve to the integer result key."""
    req = GroundednessRequirement()

    result = req._parse_batch_support_output(
        json.dumps([{"span_id": span_id, "support_level": "FULLY_SUPPORTED"}]), 1
    )

    assert result == {0: "FULLY_SUPPORTED"}


def test_parse_necessity_output_normalises_model_types():
    """Necessity parsing must tolerate quoted span IDs and string/integer labels."""
    req = GroundednessRequirement()
    spans = [
        {"begin": 0, "end": 5, "text": "Fact."},
        {"begin": 7, "end": 13, "text": "Other."},
    ]

    result = req._parse_necessity_output(
        '[{"span_id": "0", "needs_citation": "yes"}, '
        '{"span_id": 1, "needs_citation": 1}]',
        spans,
    )

    assert result == {(0, 5): True, (7, 13): True}


@pytest.mark.parametrize("needs_citation", [False, 0, "false", "0", "no"])
def test_parse_necessity_output_preserves_false_labels(needs_citation: object):
    """False model-returned necessity labels must remain false."""
    req = GroundednessRequirement()
    spans = [{"begin": 0, "end": 5, "text": "Fact."}]

    result = req._parse_necessity_output(
        json.dumps([{"span_id": 0, "needs_citation": needs_citation}]), spans
    )

    assert result == {(0, 5): False}


@pytest.mark.parametrize(
    "output_text",
    [
        (
            '[{"span_id": 0, "needs_citation": "no"}, '
            '{"span_id": "0", "needs_citation": "yes"}]'
        ),
        (
            '[{"span_id": "0", "needs_citation": "yes"}, '
            '{"span_id": 0, "needs_citation": "no"}]'
        ),
    ],
)
def test_parse_necessity_output_duplicate_ids_require_citations(output_text: str):
    """Duplicate span judgments must retain the conservative result."""
    req = GroundednessRequirement()
    spans = [{"begin": 0, "end": 5, "text": "Fact."}]

    result = req._parse_necessity_output(output_text, spans)

    assert result == {(0, 5): True}


@pytest.mark.parametrize(
    "output_text",
    [
        (
            '[{"span_id": 0, "citations": [{"support_level": "NOT_SUPPORTED"}]}, '
            '{"span_id": "0", "citations": [{"support_level": "FULLY_SUPPORTED"}]}]'
        ),
        (
            '[{"span_id": "0", "citations": [{"support_level": "FULLY_SUPPORTED"}]}, '
            '{"span_id": 0, "citations": [{"support_level": "NOT_SUPPORTED"}]}]'
        ),
    ],
)
def test_parse_batch_support_output_duplicate_ids_are_pessimistic(output_text: str):
    """Nested duplicate span judgments must retain the least-supported result."""
    req = GroundednessRequirement()

    result = req._parse_batch_support_output(output_text, 1)

    assert result == {0: "NOT_SUPPORTED"}


@pytest.mark.parametrize(
    "output_text",
    [
        (
            '[{"span_id": 0, "support_level": "NOT_SUPPORTED"}, '
            '{"span_id": "0", "support_level": "FULLY_SUPPORTED"}]'
        ),
        (
            '[{"span_id": "0", "support_level": "FULLY_SUPPORTED"}, '
            '{"span_id": 0, "support_level": "NOT_SUPPORTED"}]'
        ),
    ],
)
def test_parse_batch_support_output_duplicate_flat_ids_are_pessimistic(
    output_text: str,
):
    """Duplicate flat rows must retain the least-supported result."""
    req = GroundednessRequirement()

    result = req._parse_batch_support_output(output_text, 1)

    assert result == {0: "NOT_SUPPORTED"}


@pytest.mark.parametrize(
    "output_text",
    [
        (
            '[{"span_id": 0, "support_level": "FULLY_SUPPORTED"}, '
            '{"span_id": 0, "evidence": [{"support_level": "NOT_SUPPORTED"}]}]'
        ),
        (
            '[{"span_id": 0, "evidence": [{"support_level": "NOT_SUPPORTED"}]}, '
            '{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]'
        ),
    ],
)
def test_parse_batch_support_output_mixed_duplicate_ids_are_pessimistic(
    output_text: str,
):
    """Mixed output shapes must retain the least-supported result."""
    req = GroundednessRequirement()

    result = req._parse_batch_support_output(output_text, 1)

    assert result == {0: "NOT_SUPPORTED"}


@pytest.mark.parametrize(
    "support_level",
    [
        "NOT FULLY SUPPORTED",
        "FULLY UNSUPPORTED",
        "NOT_FULLY_SUPPORTED",
        "FULLY_UNSUPPORTED",
        "not_fully_supported",
        "NotFullySupported",
        "notFullySupported",
    ],
)
def test_parse_batch_support_output_negated_labels_are_not_supported(
    support_level: str,
):
    """Negated support labels must not be classified as fully supported."""
    req = GroundednessRequirement()

    result = req._parse_batch_support_output(
        json.dumps([{"span_id": 0, "support_level": support_level}]), 1
    )

    assert result == {0: "NOT_SUPPORTED"}


def test_parse_batch_support_output_positive_unambiguous_label_is_supported():
    """Positive labels containing unrelated `UN` text must remain supported."""
    req = GroundednessRequirement()

    result = req._parse_batch_support_output(
        '[{"span_id": 0, "support_level": "FULLY SUPPORTED (UNAMBIGUOUS)"}]', 1
    )

    assert result == {0: "FULLY_SUPPORTED"}


def test_parse_batch_support_output_nested_negated_label_is_not_supported():
    """Negated nested support labels must not be classified as fully supported."""
    req = GroundednessRequirement()

    result = req._parse_batch_support_output(
        '[{"span_id": 0, "evidence": [{"support_level": "not fully supported"}]}]', 1
    )

    assert result == {0: "NOT_SUPPORTED"}


def test_malformed_necessity_label_unsupported_span_is_errored():
    """An unreadable necessity label must not let an unsupported span pass.

    The span is neither confirmed as needing a citation nor confirmed as
    supported, so the result is errored rather than a pass or an ordinary failure.
    """
    req = GroundednessRequirement()
    spans = [{"begin": 0, "end": 5, "text": "Fact."}]
    necessity = req._parse_necessity_output(
        '[{"span_id": 0, "needs_citation": "maybe"}]', spans
    )
    assert necessity == {(0, 5): None}

    result = req._build_groundedness_result(
        "Fact.", [], necessity, {(0, 5): "NOT_SUPPORTED"}
    )

    assert result.as_bool() is False
    assert result.reason is None
    assert result.error is not None


def test_necessity_true_unsupported_span_is_ordinary_failure():
    """A span known to need a citation and not supported is an ordinary failure."""
    req = GroundednessRequirement()

    result = req._build_groundedness_result(
        "Fact.", [], {(0, 5): True}, {(0, 5): "NOT_SUPPORTED"}
    )

    assert result.as_bool() is False
    assert result.error is None
    assert result.reason is not None
    assert "Fact." in result.reason


def test_parse_batch_support_output_nested_citations():
    """Test parsing batch support output when model nests support_level inside citations.

    granite-4.0-micro frequently mirrors the input span structure from the prompt
    and nests support_level inside a 'citations' array instead of using flat format.
    See issue #1291 for deep analysis.
    """
    req = GroundednessRequirement()

    output_text = """
    [
        {
            "span_id": 0,
            "text": "The Eiffel Tower is located in Paris, France.",
            "citations": [
                {
                    "citation_id": 0,
                    "source_document": 0,
                    "support_level": "FULLY_SUPPORTED"
                }
            ]
        },
        {
            "span_id": 1,
            "text": "Another span.",
            "citations": [
                {
                    "citation_id": 0,
                    "source_document": 0,
                    "support_level": "NOT_SUPPORTED"
                }
            ]
        }
    ]
    """

    result = req._parse_batch_support_output(output_text, 2)

    assert len(result) == 2
    assert result[0] == "FULLY_SUPPORTED"
    assert result[1] == "NOT_SUPPORTED"


def test_parse_batch_support_output_nested_evidence():
    """Same as the 'citations' nesting test, but for the current 'evidence' key."""
    req = GroundednessRequirement()

    output_text = """
    [
        {
            "span_id": 0,
            "text": "The Eiffel Tower is located in Paris, France.",
            "evidence": [
                {
                    "citation_id": 0,
                    "source_document": 0,
                    "support_level": "FULLY_SUPPORTED"
                }
            ]
        },
        {
            "span_id": 1,
            "text": "Another span.",
            "evidence": [
                {
                    "citation_id": 0,
                    "source_document": 0,
                    "support_level": "NOT_SUPPORTED"
                }
            ]
        }
    ]
    """

    result = req._parse_batch_support_output(output_text, 2)

    assert len(result) == 2
    assert result[0] == "FULLY_SUPPORTED"
    assert result[1] == "NOT_SUPPORTED"


def test_parse_batch_support_output_nested_both_keys_union():
    """Both keys present at once must union, not pick one and drop the other."""
    req = GroundednessRequirement()

    output_text = """
    [
        {
            "span_id": 0,
            "evidence": [{"support_level": "FULLY_SUPPORTED"}],
            "citations": [{"support_level": "NOT_SUPPORTED"}]
        }
    ]
    """

    result = req._parse_batch_support_output(output_text, 1)

    assert len(result) == 1
    # pessimistic: NOT_SUPPORTED wins over FULLY_SUPPORTED from the other key
    assert result[0] == "NOT_SUPPORTED"


def test_parse_batch_support_output_nested_non_list_value():
    """A non-list value must not raise - degrade to unknown (None) instead."""
    req = GroundednessRequirement()

    output_text = """
    [
        {"span_id": 0, "evidence": 5},
        {"span_id": 1, "citations": "not a list"}
    ]
    """

    result = req._parse_batch_support_output(output_text, 2)

    assert result == {0: None, 1: None}


def test_parse_batch_support_output_nested_citations_pessimistic_aggregate():
    """Mixed nested citation support levels on one span aggregate pessimistically.

    NOT_SUPPORTED beats PARTIALLY_SUPPORTED beats FULLY_SUPPORTED, regardless
    of citation order in the LLM response.
    """
    req = GroundednessRequirement()

    output_text = """
    [
        {
            "span_id": 0,
            "text": "Mixed-support span.",
            "citations": [
                {"citation_id": 0, "source_document": 0, "support_level": "FULLY_SUPPORTED"},
                {"citation_id": 1, "source_document": 0, "support_level": "NOT_SUPPORTED"},
                {"citation_id": 2, "source_document": 0, "support_level": "PARTIALLY_SUPPORTED"}
            ]
        }
    ]
    """

    result = req._parse_batch_support_output(output_text, 1)

    assert len(result) == 1
    assert result[0] == "NOT_SUPPORTED"


def test_parse_batch_support_output_nested_multiple_citations():
    """Span with multiple nested citations of mixed levels -> pessimistic pick.

    This exercises the pessimistic aggregation loop itself (the existing test
    only has one citation per span, so the ordering never has to choose).
    """
    req = GroundednessRequirement()

    output_text = """
    [
        {
            "span_id": 0,
            "citations": [
                {"support_level": "FULLY_SUPPORTED"},
                {"support_level": "NOT_SUPPORTED"}
            ]
        },
        {"span_id": 1, "citations": []}
    ]
    """

    result = req._parse_batch_support_output(output_text, 2)

    assert len(result) == 2
    # NOT_SUPPORTED is pessimistically chosen over FULLY_SUPPORTED.
    assert result[0] == "NOT_SUPPORTED"
    # Empty citations list is unknown.
    assert result[1] is None


def test_parse_batch_support_output_nested_label_variants():
    """Near-miss labels (space instead of underscore) must normalize, not downgrade."""
    req = GroundednessRequirement()

    output_text = """
    [
        {"span_id": 0, "citations": [{"support_level": "FULLY SUPPORTED"}]}
    ]
    """

    result = req._parse_batch_support_output(output_text, 1)

    assert result[0] == "FULLY_SUPPORTED"


def test_parse_batch_support_output_with_recovery():
    """Malformed JSON is repaired up to the last complete object.

    The truncated trailing object cannot be recovered, so its span is unknown.
    """
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

    assert set(result) == {0, 1}
    assert result[0] == "FULLY_SUPPORTED"
    assert result[1] is None


def test_parse_batch_support_output_missing_spans():
    """A span missing from the output is unknown, not NOT_SUPPORTED."""
    req = GroundednessRequirement()

    # Output only has one span, but we expect two
    output_text = '[{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]'

    result = req._parse_batch_support_output(output_text, 2)

    assert result == {0: "FULLY_SUPPORTED", 1: None}


def test_parse_batch_support_output_invalid_json():
    """Unreadable output makes every expected index unknown."""
    req = GroundednessRequirement()

    # Invalid JSON
    output_text = "This is not JSON at all"

    result = req._parse_batch_support_output(output_text, 2)

    assert result == {0: None, 1: None}


# ---------------------------------------------------------------------------
# Errors-not-verdicts tests (#1718). All use a mocked AdapterMixin backend.
# ---------------------------------------------------------------------------

_FULLY = "FULLY_SUPPORTED"
_PARTIAL = "PARTIALLY_SUPPORTED"
_NOT = "NOT_SUPPORTED"

_CITATION = {
    "response_begin": 0,
    "response_end": 9,
    "citation_text": "Fact one",
    "citation_doc_id": "0",
}
_TWO_FACTS = "Fact one. Fact two."
_NECESSITY_BOTH = (
    '[{"span_id": 0, "needs_citation": "yes"}, {"span_id": 1, "needs_citation": "yes"}]'
)
_ORIGINAL = RuntimeError("adapter boom")


def _thunk(value: str | None) -> MagicMock:
    thunk = MagicMock(spec=ModelOutputThunk)
    thunk.avalue = AsyncMock()
    thunk.value = value
    return thunk


def _adapter_backend(*responses: object) -> MagicMock:
    """Build a mock AdapterMixin backend.

    Each response is either a string/None (the judged model output) or an
    exception instance (raised by `generate_from_context`).
    """
    backend = MagicMock(spec=AdapterMixin)
    side_effect: list[object] = []
    for response in responses:
        if isinstance(response, BaseException):
            side_effect.append(response)
        else:
            assert response is None or isinstance(response, str)
            side_effect.append((_thunk(response), ChatContext()))
    backend.generate_from_context = AsyncMock(side_effect=side_effect)
    return backend


def _ctx(response: str = _TWO_FACTS) -> ChatContext:
    return (
        ChatContext()
        .add(Message("user", "question"))
        .add(Message("assistant", response))
    )


def _req(**kwargs: object) -> GroundednessRequirement:
    return GroundednessRequirement(documents=["Fact one is in a document."], **kwargs)  # type: ignore[arg-type]


def _patch_citations(citations: list[dict] | BaseException):
    if isinstance(citations, BaseException):
        return patch(
            "mellea.stdlib.components.intrinsic.rag.find_citations",
            side_effect=citations,
        )
    return patch(
        "mellea.stdlib.components.intrinsic.rag.find_citations", return_value=citations
    )


def _assert_errored(result) -> None:
    assert result.as_bool() is False
    assert result.reason is None
    assert result.error is not None


# --- Setup checks ----------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_empty_context_is_errored():
    """Empty context is a setup problem, not a verdict."""
    result = await _req().validate(_adapter_backend(), ChatContext())
    _assert_errored(result)
    assert "Context is empty" in str(result.error)


@pytest.mark.asyncio
async def test_setup_last_item_not_message_is_errored():
    """A non-Message last item is a setup problem."""
    ctx = ChatContext().add(Message("user", "q"))
    with patch.object(ChatContext, "as_list", return_value=[MagicMock()]):
        result = await _req().validate(_adapter_backend(), ctx)
    _assert_errored(result)
    assert "not a Message" in str(result.error)


@pytest.mark.asyncio
async def test_setup_last_message_not_assistant_is_errored():
    """A non-assistant last message is a setup problem."""
    ctx = ChatContext().add(Message("user", "q"))
    result = await _req().validate(_adapter_backend(), ctx)
    _assert_errored(result)
    assert "got role: user" in str(result.error)


@pytest.mark.asyncio
async def test_setup_no_documents_is_errored():
    """Missing documents is a setup problem."""
    result = await GroundednessRequirement().validate(_adapter_backend(), _ctx())
    _assert_errored(result)
    assert "No documents provided" in str(result.error)


@pytest.mark.asyncio
async def test_setup_backend_without_adapters_is_errored():
    """A backend that is not an AdapterMixin is a setup problem."""
    result = await _req().validate(MagicMock(), _ctx())
    _assert_errored(result)
    assert "does not support adapters" in str(result.error)


@pytest.mark.asyncio
async def test_empty_response_still_passes():
    """The empty-response pass is a real verdict and is unchanged."""
    result = await _req().validate(_adapter_backend(), _ctx(""))
    assert result.as_bool() is True
    assert result.error is None


# --- Stage handlers --------------------------------------------------------


@pytest.mark.asyncio
async def test_stage1_citation_failure_preserves_original_exception(caplog):
    """Stage 1 is not wrapped: `error` is the original exception."""
    with _patch_citations(_ORIGINAL), caplog.at_level(logging.ERROR, logger="mellea"):
        result = await _req().validate(_adapter_backend(), _ctx())
    _assert_errored(result)
    assert result.error is _ORIGINAL
    assert "Citation generation adapter function failed" in caplog.text


@pytest.mark.asyncio
async def test_stage2_backend_failure_is_wrapped(caplog):
    """Stage 2 wraps a model-call failure; the original is on `__context__`."""
    backend = _adapter_backend(_ORIGINAL)
    with _patch_citations([]), caplog.at_level(logging.ERROR, logger="mellea"):
        result = await _req().validate(backend, _ctx())
    _assert_errored(result)
    assert isinstance(result.error, ValueError)
    assert str(result.error).startswith("LLM judgment failed:")
    assert result.error.__context__ is _ORIGINAL
    assert "Citation necessity assessment failed" in caplog.text


@pytest.mark.asyncio
async def test_stage2_none_output_is_wrapped():
    """Stage 2 `None` output: the inner error is wrapped again."""
    backend = _adapter_backend(None)
    with _patch_citations([]):
        result = await _req().validate(backend, _ctx())
    _assert_errored(result)
    assert str(result.error).startswith("LLM judgment failed:")
    assert isinstance(result.error.__context__, ValueError)
    assert "returned None" in str(result.error.__context__)


@pytest.mark.asyncio
async def test_stage2_parser_failure_is_unwrapped():
    """A parser exception is raised outside the wrapper and arrives as is."""
    backend = _adapter_backend("[]")
    with (
        _patch_citations([]),
        patch.object(
            GroundednessRequirement, "_parse_necessity_output", side_effect=_ORIGINAL
        ),
    ):
        result = await _req().validate(backend, _ctx())
    _assert_errored(result)
    assert result.error is _ORIGINAL


@pytest.mark.asyncio
async def test_stage3_backend_failure_is_wrapped(caplog):
    """Stage 3 wraps a model-call failure; the original is on `__context__`."""
    backend = _adapter_backend(_NECESSITY_BOTH, _ORIGINAL)
    with _patch_citations([_CITATION]), caplog.at_level(logging.ERROR, logger="mellea"):
        result = await _req().validate(backend, _ctx())
    _assert_errored(result)
    assert str(result.error).startswith("Batch LLM judgment for support failed:")
    assert result.error.__context__ is _ORIGINAL
    assert "Citation support assessment failed" in caplog.text


@pytest.mark.asyncio
async def test_stage3_none_output_is_wrapped():
    """Stage 3 `None` output: the inner error is wrapped again."""
    backend = _adapter_backend(_NECESSITY_BOTH, None)
    with _patch_citations([_CITATION]):
        result = await _req().validate(backend, _ctx())
    _assert_errored(result)
    assert str(result.error).startswith("Batch LLM judgment for support failed:")
    assert isinstance(result.error.__context__, ValueError)
    assert "returned None" in str(result.error.__context__)


@pytest.mark.asyncio
async def test_stage3_parser_failure_is_unwrapped():
    """A support-parser exception is raised outside the wrapper."""
    backend = _adapter_backend(_NECESSITY_BOTH, "[]")
    with (
        _patch_citations([_CITATION]),
        patch.object(
            GroundednessRequirement,
            "_parse_batch_support_output",
            side_effect=_ORIGINAL,
        ),
    ):
        result = await _req().validate(backend, _ctx())
    _assert_errored(result)
    assert result.error is _ORIGINAL


@pytest.mark.asyncio
async def test_step4_failure_is_errored_and_logged(caplog):
    """An exception from the verdict builder becomes an errored result."""
    backend = _adapter_backend(
        _NECESSITY_BOTH, '[{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]'
    )
    with (
        _patch_citations([_CITATION]),
        patch.object(
            GroundednessRequirement, "_build_groundedness_result", side_effect=_ORIGINAL
        ),
        caplog.at_level(logging.ERROR, logger="mellea"),
    ):
        result = await _req().validate(backend, _ctx())
    _assert_errored(result)
    assert result.error is _ORIGINAL
    assert "Groundedness result construction failed" in caplog.text


@pytest.mark.asyncio
async def test_full_pipeline_grounded_passes():
    """Happy path through all four steps still passes."""
    backend = _adapter_backend(
        '[{"span_id": 0, "needs_citation": "yes"}, {"span_id": 1, "needs_citation": "no"}]',
        '[{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]',
    )
    with _patch_citations([_CITATION]):
        result = await _req().validate(backend, _ctx())
    assert result.as_bool() is True
    assert result.error is None
    assert "1/1" in (result.reason or "")


# --- Support parser: unreadable input gives unknown -------------------------


@pytest.mark.parametrize(
    "output_text",
    [
        "This is not JSON at all",
        "no bracket here",
        '[{"span_id": 0, "support',
        '{"span_id": 0, "support_level": "FULLY_SUPPORTED"}',
        "5",
        '[1, "a"]',
        '[[{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]]',
    ],
)
def test_support_parser_unreadable_input_is_all_unknown(output_text: str):
    """Unreadable or wrongly shaped output gives `None` for every index.

    A bare object with no bracket raises inside `_extract_and_repair_json_array`
    and is classified unreadable. Non-dict entries are skipped, so every span is
    missing.
    """
    result = GroundednessRequirement()._parse_batch_support_output(output_text, 2)
    assert result == {0: None, 1: None}


def test_support_parser_object_wrapping_array_is_read():
    """An object wrapping an array is read through the existing bracket search."""
    result = GroundednessRequirement()._parse_batch_support_output(
        '{"results": [{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]}', 1
    )
    assert result == {0: _FULLY}


@pytest.mark.parametrize(
    "judgment",
    [
        '{"span_id": 0, "support_level": ""}',
        '{"span_id": 0, "support_level": "maybe"}',
        '{"span_id": 0, "support_level": 7}',
        '{"span_id": 0}',
    ],
)
def test_support_parser_unreadable_label_is_unknown_for_that_index_only(judgment: str):
    """Empty, unrecognised and non-string labels give `None` for that index only."""
    result = GroundednessRequirement()._parse_batch_support_output(
        f'[{judgment}, {{"span_id": 1, "support_level": "FULLY_SUPPORTED"}}]', 2
    )
    assert result == {0: None, 1: _FULLY}


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ([_FULLY, "maybe"], None),
        ([_FULLY, _NOT, "maybe"], _NOT),
        ([_FULLY, _PARTIAL], _PARTIAL),
        ([_FULLY, _FULLY], _FULLY),
        ([], None),
        ([5], None),
    ],
)
def test_support_parser_nested_aggregation(evidence: list, expected: str | None):
    """Nested judgments reduce: NOT wins, then any unreadable, then most pessimistic."""
    nested = [{"support_level": e} if isinstance(e, str) else e for e in evidence]
    output = json.dumps([{"span_id": 0, "evidence": nested}])
    result = GroundednessRequirement()._parse_batch_support_output(output, 1)
    assert result == {0: expected}


def test_support_parser_non_dict_entry_in_evidence_is_unknown():
    """A non-dict entry inside `evidence` counts as an unreadable judgment."""
    output = json.dumps(
        [{"span_id": 0, "evidence": [{"support_level": _FULLY}, "text"]}]
    )
    result = GroundednessRequirement()._parse_batch_support_output(output, 1)
    assert result == {0: None}


def test_support_parser_non_list_evidence_is_unknown():
    """A non-list `evidence` value gives `None`."""
    result = GroundednessRequirement()._parse_batch_support_output(
        '[{"span_id": 0, "evidence": "text"}]', 1
    )
    assert result == {0: None}


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        ([_FULLY, "maybe"], None),
        (["maybe", _FULLY], None),
        ([_FULLY, _PARTIAL], _PARTIAL),
        ([_NOT, "maybe"], _NOT),
    ],
)
def test_support_parser_duplicate_judgments_follow_same_contract(
    levels: list[str], expected: str | None
):
    """Duplicate top-level judgments reduce like nested labels."""
    output = json.dumps([{"span_id": 0, "support_level": level} for level in levels])
    result = GroundednessRequirement()._parse_batch_support_output(output, 1)
    assert result == {0: expected}


def test_support_parser_does_not_swallow_internal_errors():
    """Anything raised inside the parser (not the JSON extraction) propagates."""
    with (
        patch(
            "mellea.stdlib.requirements.rag._normalise_span_id",
            side_effect=RuntimeError("bug"),
        ),
        pytest.raises(RuntimeError, match="bug"),
    ):
        GroundednessRequirement()._parse_batch_support_output(
            '[{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]', 1
        )


@pytest.mark.parametrize("n", [0, 1, 3])
def test_support_parser_returns_complete_mapping(n: int):
    """Keys are exactly `range(n)` for unreadable and partial output."""
    req = GroundednessRequirement()
    assert set(req._parse_batch_support_output("garbage", n)) == set(range(n))
    assert set(
        req._parse_batch_support_output(
            '[{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]', n
        )
    ) == set(range(n))


# --- Necessity parser -------------------------------------------------------


_ONE_SPAN = [{"begin": 0, "end": 5, "text": "Fact."}]


@pytest.mark.parametrize(
    "output_text", ["garbage", "no bracket", "[1, 2]", '[{"span_id": 5}]', "[]"]
)
def test_necessity_parser_unreadable_or_missing_is_unknown(output_text: str):
    """Unreadable output and missing spans give `None` for every span key."""
    result = GroundednessRequirement()._parse_necessity_output(output_text, _ONE_SPAN)
    assert result == {(0, 5): None}


@pytest.mark.parametrize("value", ["maybe", 2.5, [1], {"a": 1}, 2, -1, None])
def test_necessity_parser_unrecognised_value_is_unknown(value: object):
    """Unrecognised strings, floats, lists, dicts, other integers and null give `None`."""
    result = GroundednessRequirement()._parse_necessity_output(
        json.dumps([{"span_id": 0, "needs_citation": value}]), _ONE_SPAN
    )
    assert result == {(0, 5): None}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (2, None),
        (-1, None),
        ("Yes", True),
        (" true ", True),
        ("1", True),
        ("no", False),
        ("0", False),
        ("maybe", None),
        (1.0, None),
        (None, None),
        ([], None),
    ],
)
def test_normalise_needs_citation(raw: object, expected: bool | None):
    """Only recognised values map to a bool; `bool` is checked before `int`."""
    assert _normalise_needs_citation(raw) is expected


@pytest.mark.parametrize("values", list(itertools.permutations([True, "maybe", False])))
def test_necessity_parser_duplicates_true_beats_unknown_beats_false(values: tuple):
    """`{True, None, False}` in every order gives `True`."""
    output = json.dumps([{"span_id": 0, "needs_citation": v} for v in values])
    result = GroundednessRequirement()._parse_necessity_output(output, _ONE_SPAN)
    assert result == {(0, 5): True}


@pytest.mark.parametrize(
    ("values", "expected"),
    [(["maybe", False], None), ([False, "maybe"], None), ([False, False], False)],
)
def test_necessity_parser_duplicate_merge_without_true(
    values: list, expected: bool | None
):
    """`{None, False}` gives `None`; `{False, False}` gives `False`."""
    output = json.dumps([{"span_id": 0, "needs_citation": v} for v in values])
    result = GroundednessRequirement()._parse_necessity_output(output, _ONE_SPAN)
    assert result == {(0, 5): expected}


def test_necessity_parser_returns_exactly_input_span_keys():
    """Every input span is a key, whatever the output looks like."""
    spans = [{"begin": 0, "end": 5, "text": "a"}, {"begin": 7, "end": 9, "text": "b"}]
    result = GroundednessRequirement()._parse_necessity_output("garbage", spans)
    assert result == {(0, 5): None, (7, 9): None}


def test_necessity_parser_does_not_swallow_internal_errors():
    """Anything raised inside the parser (not the JSON extraction) propagates."""
    with (
        patch(
            "mellea.stdlib.requirements.rag._normalise_span_id",
            side_effect=RuntimeError("bug"),
        ),
        pytest.raises(RuntimeError, match="bug"),
    ):
        GroundednessRequirement()._parse_necessity_output(
            '[{"span_id": 0, "needs_citation": true}]', _ONE_SPAN
        )


# --- Verdict builder --------------------------------------------------------

_PASS, _FAIL, _ERRORED = "pass", "fail", "errored"


@pytest.mark.parametrize("allow_partial", [False, True])
@pytest.mark.parametrize(
    ("necessity", "support", "partial_pass", "partial_fail"),
    [
        (True, _NOT, _FAIL, _FAIL),
        (True, _PARTIAL, _PASS, _FAIL),
        (True, _FULLY, _PASS, _PASS),
        (True, None, _ERRORED, _ERRORED),
        (None, _NOT, _ERRORED, _ERRORED),
        (None, _PARTIAL, _PASS, _ERRORED),
        (None, _FULLY, _PASS, _PASS),
        (False, None, _PASS, _PASS),
    ],
)
def test_verdict_boundary_table(
    allow_partial: bool,
    necessity: bool | None,
    support: str | None,
    partial_pass: str,
    partial_fail: str,
):
    """Pin every row of the boundary table for both `allow_partial_support` values.

    The outcome is `(as_bool(), reason is None, error is None)`.
    """
    req = GroundednessRequirement(allow_partial_support=allow_partial)
    expected = partial_pass if allow_partial else partial_fail

    result = req._build_groundedness_result(
        "Fact.", [], {(0, 5): necessity}, {(0, 5): support}
    )

    outcome = (result.as_bool(), result.reason is None, result.error is None)
    assert (
        outcome
        == {
            _PASS: (True, False, True),
            _FAIL: (False, False, True),
            _ERRORED: (False, True, False),
        }[expected]
    )


def test_verdict_failure_names_only_failing_spans():
    """A determined failure wins over an unknown span and names only failing spans."""
    response = "Alpha. Beta."
    result = GroundednessRequirement()._build_groundedness_result(
        response, [], {(0, 6): True, (6, 12): True}, {(0, 6): _NOT, (6, 12): None}
    )
    assert result.as_bool() is False
    assert result.error is None
    assert result.reason is not None
    assert "Alpha." in result.reason
    assert "Beta." not in result.reason


def test_verdict_known_pass_plus_unknown_is_errored():
    """An unknown span never lets a result pass."""
    result = GroundednessRequirement()._build_groundedness_result(
        "Alpha. Beta.",
        [],
        {(0, 6): True, (6, 12): True},
        {(0, 6): _FULLY, (6, 12): None},
    )
    _assert_errored(result)
    assert "Beta." in str(result.error)
    assert "Alpha." not in str(result.error)
    assert "1 span(s)" in str(result.error)


def test_verdict_unread_necessity_unsupported_is_errored_not_failure():
    """Unread necessity plus `NOT_SUPPORTED` is errored."""
    result = GroundednessRequirement()._build_groundedness_result(
        "Fact.", [], {(0, 5): None}, {(0, 5): _NOT}
    )
    _assert_errored(result)


def test_verdict_unread_necessity_fully_supported_passes():
    """Unread necessity plus `FULLY_SUPPORTED` is grounded either way."""
    result = GroundednessRequirement()._build_groundedness_result(
        "Fact.", [], {(0, 5): None}, {(0, 5): _FULLY}
    )
    assert result.as_bool() is True


def test_verdict_unread_necessity_partial_support_passes_when_partial_allowed():
    """Unread necessity plus `PARTIALLY_SUPPORTED` passes when partial support is allowed."""
    result = GroundednessRequirement(
        allow_partial_support=True
    )._build_groundedness_result("Fact.", [], {(0, 5): None}, {(0, 5): _PARTIAL})
    assert result.as_bool() is True


def test_verdict_unread_necessity_partial_support_errored_when_partial_disallowed():
    """Unread necessity plus `PARTIALLY_SUPPORTED` is errored when partial support is not allowed."""
    result = GroundednessRequirement(
        allow_partial_support=False
    )._build_groundedness_result("Fact.", [], {(0, 5): None}, {(0, 5): _PARTIAL})
    _assert_errored(result)


def test_verdict_error_message_truncates_long_span_lists():
    """The error quotes at most five spans and counts the rest."""
    response = "x" * 80
    necessity = {(i, i + 1): None for i in range(7)}
    support = dict.fromkeys(necessity)
    result = GroundednessRequirement()._build_groundedness_result(
        response, [], necessity, support
    )
    assert "7 span(s)" in str(result.error)
    assert "(and 2 more)" in str(result.error)


# --- Invariants -------------------------------------------------------------


@pytest.mark.parametrize("necessity", [True, None])
def test_invariant_missing_support_entry_raises(necessity: bool | None):
    """A span with necessity True/None and no support entry is an invariant violation."""
    with pytest.raises(ValueError, match=r"\(0, 5\)"):
        GroundednessRequirement()._build_groundedness_result(
            "Fact.", [], {(0, 5): necessity}, {}
        )


@pytest.mark.parametrize("necessity", [True, None])
def test_invariant_unrecognised_support_value_raises(necessity: bool | None):
    """A support value that is neither None nor a known level raises."""
    with pytest.raises(ValueError, match=r"\(0, 5\)"):
        GroundednessRequirement()._build_groundedness_result(
            "Fact.", [], {(0, 5): necessity}, {(0, 5): "SORT_OF_SUPPORTED"}
        )


def test_invariant_support_key_without_necessity_raises():
    """Support for a span absent from `span_necessity` is an invariant violation."""
    with pytest.raises(ValueError, match=r"\(9, 9\)"):
        GroundednessRequirement()._build_groundedness_result(
            "Fact.", [], {(0, 0): True}, {(0, 0): _FULLY, (9, 9): _FULLY}
        )


@pytest.mark.parametrize("stray", [_NOT, "NONSENSE", None])
def test_stray_support_entry_for_false_necessity_is_ignored(stray: str | None):
    """A present entry for a `False`-necessity span is never read."""
    result = GroundednessRequirement()._build_groundedness_result(
        "Fact.", [], {(0, 5): False}, {(0, 5): stray}
    )
    assert result.as_bool() is True
    assert "Fact." not in (result.reason or "")
    assert "0/0" in (result.reason or "")


def test_false_necessity_span_with_no_entry_passes():
    """`False`-necessity spans need no support entry."""
    result = GroundednessRequirement()._build_groundedness_result(
        "Fact.", [], {(0, 5): False}, {}
    )
    assert result.as_bool() is True


# --- Summary line -----------------------------------------------------------


def test_summary_denominator_counts_only_true_necessity_spans():
    """3 True spans + 2 None spans, 2 True spans fully supported gives `2/3`."""
    response = "a" * 40
    necessity = {(0, 1): True, (1, 2): True, (2, 3): True, (3, 4): None, (4, 5): None}
    support = {
        (0, 1): _FULLY,
        (1, 2): _FULLY,
        (2, 3): _PARTIAL,
        (3, 4): _FULLY,
        (4, 5): _FULLY,
    }
    result = GroundednessRequirement(
        allow_partial_support=True
    )._build_groundedness_result(response, [], necessity, support)
    assert result.as_bool() is True
    assert "Summary: 2/3 " in (result.reason or "")


def test_summary_on_failure_and_absent_from_error():
    """The summary is on failure reasons and never in an errored `error` message."""
    req = GroundednessRequirement()
    failed = req._build_groundedness_result("Fact.", [], {(0, 5): True}, {(0, 5): _NOT})
    assert "Summary: 0/1" in (failed.reason or "")

    errored = req._build_groundedness_result(
        "Fact.", [], {(0, 5): True}, {(0, 5): None}
    )
    assert errored.reason is None
    assert "Summary" not in str(errored.error)


def test_summary_numerator_counts_fully_supported_not_policy_passes():
    """With partial allowed, a partially supported span passes but is not in the numerator."""
    result = GroundednessRequirement(
        allow_partial_support=True
    )._build_groundedness_result(
        "Alpha. Beta.",
        [],
        {(0, 6): True, (6, 12): True},
        {(0, 6): _FULLY, (6, 12): _PARTIAL},
    )
    assert result.as_bool() is True
    assert "Summary: 1/2 spans needing citations are fully supported." in (
        result.reason or ""
    )


@pytest.mark.parametrize(
    ("necessity", "support"),
    [({}, {}), ({(0, 5): False}, {}), ({(0, 5): None}, {(0, 5): _FULLY})],
)
def test_summary_empty_denominator_does_not_raise(necessity: dict, support: dict):
    """With no `True` spans the summary reads `0/0` and cannot raise."""
    result = GroundednessRequirement()._build_groundedness_result(
        "Fact.", [], necessity, support
    )
    assert result.as_bool() is True
    assert (result.reason or "").endswith(
        "0/0 spans needing citations are fully supported."
    )


# --- Support stage ----------------------------------------------------------


@pytest.mark.asyncio
async def test_assess_support_unknown_necessity_is_assessed():
    """A `None`-necessity span with a citation goes to the judge; one without is NOT_SUPPORTED."""
    req = GroundednessRequirement()
    response = "Fact one. Fact two. Fact three."
    citations = [
        {
            "response_begin": 0,
            "response_end": 9,
            "citation_text": "c",
            "citation_doc_id": "0",
        }
    ]
    backend = _adapter_backend(
        '[{"span_id": 0, "support_level": "PARTIALLY_SUPPORTED"}]'
    )

    support = await req._assess_citation_support(
        response,
        citations,
        {(0, 9): None, (10, 19): None, (20, 31): False},
        backend,
        ChatContext().add(Message("user", "q")),
        [],
    )

    assert support == {(0, 9): _PARTIAL, (10, 19): _NOT}
    assert backend.generate_from_context.await_count == 1


@pytest.mark.asyncio
async def test_assess_support_missing_batch_result_is_unknown():
    """A span the batch judge did not return maps to `None`, not NOT_SUPPORTED."""
    req = GroundednessRequirement()
    backend = _adapter_backend("[]")
    support = await req._assess_citation_support(
        "Fact one.",
        [_CITATION],
        {(0, 9): True},
        backend,
        ChatContext().add(Message("user", "q")),
        [],
    )
    assert support == {(0, 9): None}


# --- Token cap --------------------------------------------------------------


@pytest.mark.asyncio
async def test_judgment_calls_use_model_option_max_new_tokens():
    """Both judgment calls send `ModelOption.MAX_NEW_TOKENS` and no plain key."""
    req = GroundednessRequirement(max_new_tokens=123)
    backend = _adapter_backend(
        '[{"span_id": 0, "needs_citation": "yes"}]',
        '[{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]',
    )
    ctx = ChatContext().add(Message("user", "q"))

    necessity = await req._identify_citation_necessity(
        "Fact one.", [_CITATION], backend, ctx
    )
    await req._assess_citation_support(
        "Fact one.", [_CITATION], necessity, backend, ctx, []
    )

    assert backend.generate_from_context.await_count == 2
    for call in backend.generate_from_context.await_args_list:
        options = call.kwargs["model_options"]
        assert options[ModelOption.MAX_NEW_TOKENS] == 123
        assert "max_new_tokens" not in options


# --- Original bug, end to end with mocks -------------------------------------


def _repair_strings(req: Requirement, result: ValidationResult) -> tuple[str, str]:
    instruction = Instruction("Do the task")
    plain, _ = RepairTemplateStrategy.repair(
        ChatContext(), ChatContext(), [instruction], [None], [[(req, result)]]
    )
    friendly, _ = ModelFriendlyRepairStrategy.repair(
        ChatContext(), ChatContext(), [instruction], [None], [[(req, result)]]
    )
    assert isinstance(plain, Instruction)
    assert isinstance(friendly, Instruction)
    assert plain._repair_string is not None
    assert friendly._repair_string is not None
    return plain._repair_string, friendly._repair_string


@pytest.mark.asyncio
async def test_adapter_failure_does_not_reach_repair_prompt():
    """Stage 1 failure: the repair prompt names the requirement, not the exception."""
    req = _req()
    with _patch_citations(_ORIGINAL):
        result = await req.validate(_adapter_backend(), _ctx())
    assert result.error is _ORIGINAL

    plain, friendly = _repair_strings(req, result)

    assert req.description in plain
    assert "adapter boom" not in plain
    assert req.description in friendly
    assert "adapter boom" not in friendly
    assert "Details" not in friendly


@pytest.mark.asyncio
async def test_setup_failure_does_not_reach_repair_prompt():
    """Setup failure: the repair prompt does not carry the setup message."""
    req = GroundednessRequirement()
    result = await req.validate(_adapter_backend(), _ctx())
    assert "No documents provided" in str(result.error)

    plain, friendly = _repair_strings(req, result)

    assert "No documents provided" not in plain
    assert "No documents provided" not in friendly
    assert "Details" not in friendly


@pytest.mark.asyncio
async def test_stage2_failure_does_not_reach_repair_prompt():
    """Stage 2 failure: `error` is the wrapper with the original on `__context__`."""
    req = _req()
    with _patch_citations([]):
        result = await req.validate(_adapter_backend(_ORIGINAL), _ctx())
    assert str(result.error).startswith("LLM judgment failed:")
    assert result.error.__context__ is _ORIGINAL

    plain, friendly = _repair_strings(req, result)

    assert "adapter boom" not in plain
    assert "adapter boom" not in friendly
    assert "Details" not in friendly
