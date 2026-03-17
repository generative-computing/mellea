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
