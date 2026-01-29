"""Tests for the interop module (external LLM validation API)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.core import CBlock, ModelOutputThunk, Requirement, ValidationResult
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.interop import (
    ExternalSession,
    IVRResult,
    _build_context,
    _convert_message_to_mellea,
    _default_repair_prompt,
    _extract_output_string,
    _is_langchain_message,
    _is_openai_message,
    _normalize_requirements,
    aexternal_ivr,
    aexternal_validate,
    external_ivr,
    external_validate,
)
from mellea.stdlib.requirements import req

# Check if langchain is available
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = MagicMock()

    # Mock generate_from_context to return a computed ModelOutputThunk
    async def mock_generate(*args, **kwargs):
        thunk = ModelOutputThunk("Mock output")
        thunk._computed = True
        ctx = kwargs.get("ctx", ChatContext())
        return thunk, ctx

    backend.generate_from_context = AsyncMock(side_effect=mock_generate)
    return backend


@pytest.fixture
def openai_messages():
    """Sample OpenAI-format messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]


@pytest.fixture
def mellea_messages():
    """Sample Mellea messages."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="The answer is 4."),
    ]


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestIsOpenAIMessage:
    """Tests for _is_openai_message helper."""

    def test_valid_openai_message(self):
        """Test detection of valid OpenAI messages."""
        msg = {"role": "user", "content": "Hello"}
        assert _is_openai_message(msg) is True

    def test_missing_role(self):
        """Test that messages without role are not detected."""
        msg = {"content": "Hello"}
        assert _is_openai_message(msg) is False

    def test_non_dict(self):
        """Test that non-dict values are not detected."""
        assert _is_openai_message("hello") is False
        assert _is_openai_message(123) is False
        assert _is_openai_message(None) is False


class TestIsLangchainMessage:
    """Tests for _is_langchain_message helper."""

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
    def test_valid_langchain_message(self):
        """Test detection of valid LangChain messages."""
        msg = HumanMessage(content="Hello")
        assert _is_langchain_message(msg) is True

    def test_non_langchain(self):
        """Test that non-LangChain objects are not detected."""
        assert _is_langchain_message({"role": "user", "content": "Hello"}) is False
        assert _is_langchain_message(Message(role="user", content="Hello")) is False
        assert _is_langchain_message("hello") is False


class TestConvertMessageToMellea:
    """Tests for _convert_message_to_mellea helper."""

    def test_mellea_passthrough(self):
        """Test that Mellea messages pass through unchanged."""
        msg = Message(role="user", content="Hello")
        result = _convert_message_to_mellea(msg)
        assert result is msg

    def test_openai_conversion(self):
        """Test conversion from OpenAI format."""
        msg = {"role": "user", "content": "Hello"}
        result = _convert_message_to_mellea(msg)
        assert isinstance(result, Message)
        assert result.role == "user"
        assert result.content == "Hello"

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
    def test_langchain_conversion(self):
        """Test conversion from LangChain format."""
        msg = HumanMessage(content="Hello")
        result = _convert_message_to_mellea(msg)
        assert isinstance(result, Message)
        assert result.role == "user"
        assert result.content == "Hello"

    def test_invalid_format(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported message format"):
            _convert_message_to_mellea(12345)


class TestBuildContext:
    """Tests for _build_context helper."""

    def test_empty_context(self):
        """Test building context from None."""
        ctx = _build_context(None)
        assert isinstance(ctx, ChatContext)
        assert ctx.is_root_node

    def test_empty_list(self):
        """Test building context from empty list."""
        ctx = _build_context([])
        assert isinstance(ctx, ChatContext)
        assert ctx.is_root_node

    def test_openai_messages(self, openai_messages):
        """Test building context from OpenAI messages."""
        ctx = _build_context(openai_messages)
        assert isinstance(ctx, ChatContext)
        messages = ctx.as_list()
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"

    def test_mellea_messages(self, mellea_messages):
        """Test building context from Mellea messages."""
        ctx = _build_context(mellea_messages)
        assert isinstance(ctx, ChatContext)
        messages = ctx.as_list()
        assert len(messages) == 3

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
    def test_langchain_messages(self):
        """Test building context from LangChain messages."""
        lc_messages = [
            SystemMessage(content="System"),
            HumanMessage(content="User"),
            AIMessage(content="Assistant"),
        ]
        ctx = _build_context(lc_messages)
        assert isinstance(ctx, ChatContext)
        messages = ctx.as_list()
        assert len(messages) == 3


class TestExtractOutputString:
    """Tests for _extract_output_string helper."""

    def test_string_input(self):
        """Test extracting from plain string."""
        assert _extract_output_string("Hello") == "Hello"

    def test_openai_message(self):
        """Test extracting from OpenAI message dict."""
        msg = {"role": "assistant", "content": "Hello"}
        assert _extract_output_string(msg) == "Hello"

    def test_openai_multimodal(self):
        """Test extracting from OpenAI multimodal message."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": " Part 2"},
            ],
        }
        assert _extract_output_string(msg) == "Part 1 Part 2"

    def test_mellea_message(self):
        """Test extracting from Mellea Message."""
        msg = Message(role="assistant", content="Hello")
        assert _extract_output_string(msg) == "Hello"

    def test_model_output_thunk(self):
        """Test extracting from ModelOutputThunk."""
        thunk = ModelOutputThunk("Hello")
        assert _extract_output_string(thunk) == "Hello"

    def test_model_output_thunk_none(self):
        """Test that None value raises ValueError."""
        thunk = ModelOutputThunk(None)
        thunk._computed = False
        with pytest.raises(ValueError, match="has no value"):
            _extract_output_string(thunk)

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
    def test_langchain_message(self):
        """Test extracting from LangChain message."""
        msg = AIMessage(content="Hello")
        assert _extract_output_string(msg) == "Hello"

    def test_invalid_input(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            _extract_output_string(12345)


class TestNormalizeRequirements:
    """Tests for _normalize_requirements helper."""

    def test_string_requirements(self):
        """Test converting string requirements."""
        reqs = _normalize_requirements(["Req 1", "Req 2"])
        assert all(isinstance(r, Requirement) for r in reqs)
        assert reqs[0].description == "Req 1"
        assert reqs[1].description == "Req 2"

    def test_mixed_requirements(self):
        """Test mixed string and Requirement objects."""
        req_obj = Requirement("Req 2")
        reqs = _normalize_requirements(["Req 1", req_obj])
        assert len(reqs) == 2
        assert reqs[0].description == "Req 1"
        assert reqs[1] is req_obj


class TestDefaultRepairPrompt:
    """Tests for _default_repair_prompt helper."""

    def test_basic_repair_prompt(self):
        """Test generating a basic repair prompt."""
        req1 = Requirement("Must be polite")
        vr1 = ValidationResult(False, reason="Was rude")

        prompt = _default_repair_prompt([(req1, vr1)], "Bad output")

        assert "Must be polite" in prompt
        assert "Was rude" in prompt
        assert "Bad output" in prompt
        assert "try again" in prompt.lower()

    def test_multiple_failures(self):
        """Test repair prompt with multiple failures."""
        req1 = Requirement("Req 1")
        req2 = Requirement("Req 2")
        vr1 = ValidationResult(False)
        vr2 = ValidationResult(False, reason="Specific reason")

        prompt = _default_repair_prompt([(req1, vr1), (req2, vr2)], "Output")

        assert "Req 1" in prompt
        assert "Req 2" in prompt
        assert "Specific reason" in prompt


# =============================================================================
# External Session Tests
# =============================================================================


class TestExternalSession:
    """Tests for ExternalSession class."""

    def test_from_output_string(self, mock_backend):
        """Test creating session from string output."""
        session = ExternalSession.from_output("Test output", mock_backend)

        assert session.output == "Test output"
        assert session.backend is mock_backend
        assert isinstance(session.context, ChatContext)

    def test_from_output_with_context(self, mock_backend, openai_messages):
        """Test creating session with context."""
        # Use messages without the assistant message as context
        context = openai_messages[:-1]
        session = ExternalSession.from_output(
            "Test output", mock_backend, context=context
        )

        assert session.output == "Test output"
        # Context should have system + user + assistant (from output)
        messages = session.context.as_list()
        assert len(messages) == 3

    def test_from_openai(self, mock_backend, openai_messages):
        """Test creating session from OpenAI messages."""
        session = ExternalSession.from_openai(openai_messages, mock_backend)

        assert session.output == "The answer is 4."
        assert session.backend is mock_backend

    def test_from_openai_empty(self, mock_backend):
        """Test creating session from empty OpenAI messages."""
        session = ExternalSession.from_openai([], mock_backend)

        assert session.output is None
        assert session.context.is_root_node

    def test_from_openai_no_assistant(self, mock_backend):
        """Test creating session with no assistant message."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        session = ExternalSession.from_openai(messages, mock_backend)

        assert session.output is None

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
    def test_from_langchain(self, mock_backend):
        """Test creating session from LangChain messages."""
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="User"),
            AIMessage(content="AI Response"),
        ]
        session = ExternalSession.from_langchain(messages, mock_backend)

        assert session.output == "AI Response"

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
    def test_from_langchain_empty(self, mock_backend):
        """Test creating session from empty LangChain messages."""
        session = ExternalSession.from_langchain([], mock_backend)

        assert session.output is None

    def test_validate_no_output(self, mock_backend):
        """Test that validate raises error when no output."""
        session = ExternalSession(
            context=ChatContext(), backend=mock_backend, output=None
        )

        with pytest.raises(ValueError, match="No output to validate"):
            session.validate(["Requirement"])


# =============================================================================
# IVR Result Tests
# =============================================================================


class TestIVRResult:
    """Tests for IVRResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic IVRResult."""
        result = IVRResult(
            success=True, output="Final output", attempts=2, validation_results=[]
        )

        assert result.success is True
        assert result.output == "Final output"
        assert result.attempts == 2
        assert result.all_outputs == []

    def test_with_all_outputs(self):
        """Test IVRResult with all_outputs."""
        result = IVRResult(
            success=False,
            output="Final",
            attempts=3,
            validation_results=[],
            all_outputs=["First", "Second", "Final"],
        )

        assert len(result.all_outputs) == 3
        assert result.all_outputs[0] == "First"


# =============================================================================
# Integration Tests (with mocked validation)
# =============================================================================


class TestExternalValidate:
    """Integration tests for external_validate function."""

    @pytest.mark.asyncio
    async def test_basic_validation(self):
        """Test basic validation flow."""
        mock_backend = MagicMock()

        with patch(
            "mellea.stdlib.functional.avalidate",
            new_callable=AsyncMock,
            return_value=[ValidationResult(True)],
        ) as mock_avalidate:
            results = await aexternal_validate(
                output="Test output",
                requirements=["Must be valid"],
                backend=mock_backend,
            )

            assert len(results) == 1
            assert results[0].as_bool() is True
            mock_avalidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_context(self):
        """Test validation with context."""
        mock_backend = MagicMock()
        context = [{"role": "user", "content": "Hello"}]

        with patch(
            "mellea.stdlib.functional.avalidate",
            new_callable=AsyncMock,
            return_value=[ValidationResult(True)],
        ) as mock_avalidate:
            results = await aexternal_validate(
                output="Hi there!",
                requirements=["Must be friendly"],
                backend=mock_backend,
                context=context,
            )

            assert len(results) == 1
            # Verify context was built correctly
            call_args = mock_avalidate.call_args
            ctx = call_args.kwargs["context"]
            assert isinstance(ctx, ChatContext)


class TestExternalIVR:
    """Integration tests for external_ivr function."""

    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        """Test IVR succeeding on first attempt."""
        mock_backend = MagicMock()
        generate_calls = []

        def mock_generate(messages):
            generate_calls.append(messages)
            return "Valid output"

        with patch(
            "mellea.stdlib.functional.avalidate",
            new_callable=AsyncMock,
            return_value=[ValidationResult(True)],
        ):
            result = await aexternal_ivr(
                generate_fn=mock_generate,
                requirements=["Must be valid"],
                backend=mock_backend,
                loop_budget=3,
            )

            assert result.success is True
            assert result.attempts == 1
            assert result.output == "Valid output"
            assert len(generate_calls) == 1

    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        """Test IVR succeeding after retry."""
        mock_backend = MagicMock()
        call_count = [0]

        async def mock_validate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [ValidationResult(False, reason="First failure")]
            return [ValidationResult(True)]

        generate_calls = []

        def mock_generate(messages):
            generate_calls.append(messages)
            if len(generate_calls) == 1:
                return "Bad output"
            return "Good output"

        with patch("mellea.stdlib.functional.avalidate", side_effect=mock_validate):
            result = await aexternal_ivr(
                generate_fn=mock_generate,
                requirements=["Must be valid"],
                backend=mock_backend,
                loop_budget=3,
            )

            assert result.success is True
            assert result.attempts == 2
            assert result.output == "Good output"
            assert len(result.all_outputs) == 2

    @pytest.mark.asyncio
    async def test_failure_exhausted_budget(self):
        """Test IVR failing after exhausting budget."""
        mock_backend = MagicMock()
        attempt = [0]

        def mock_generate(messages):
            attempt[0] += 1
            return f"Attempt {attempt[0]}"

        with patch(
            "mellea.stdlib.functional.avalidate",
            new_callable=AsyncMock,
            return_value=[ValidationResult(False, reason="Always fails")],
        ):
            result = await aexternal_ivr(
                generate_fn=mock_generate,
                requirements=["Impossible requirement"],
                backend=mock_backend,
                loop_budget=3,
            )

            assert result.success is False
            assert result.attempts == 3
            assert len(result.all_outputs) == 3
            assert result.output == "Attempt 3"

    @pytest.mark.asyncio
    async def test_custom_repair_prompt(self):
        """Test IVR with custom repair prompt function."""
        mock_backend = MagicMock()
        call_count = [0]

        async def mock_validate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [ValidationResult(False)]
            return [ValidationResult(True)]

        repair_calls = []

        def custom_repair(failed_reqs, output):
            repair_calls.append((failed_reqs, output))
            return "Custom repair: Please fix the issues."

        received_messages = []

        def mock_generate(messages):
            received_messages.append(list(messages))
            return "Output"

        with patch("mellea.stdlib.functional.avalidate", side_effect=mock_validate):
            await aexternal_ivr(
                generate_fn=mock_generate,
                requirements=["Req"],
                backend=mock_backend,
                loop_budget=3,
                repair_prompt_fn=custom_repair,
            )

            assert len(repair_calls) == 1
            # Check that the repair message was added to context
            assert len(received_messages) == 2
            # Second call should have repair message in context
            last_messages = received_messages[1]
            assert any(
                "Custom repair" in m.content
                for m in last_messages
                if isinstance(m, Message)
            )


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Test that all expected exports are available."""

    def test_main_package_exports(self):
        """Test imports from main mellea package."""
        from mellea import ExternalSession, external_validate

        assert ExternalSession is not None
        assert external_validate is not None

    def test_interop_module_exports(self):
        """Test imports from interop module."""
        from mellea.stdlib.interop import (
            ExternalSession,
            IVRResult,
            aexternal_ivr,
            aexternal_validate,
            external_ivr,
            external_validate,
        )

        assert external_validate is not None
        assert aexternal_validate is not None
        assert ExternalSession is not None
        assert external_ivr is not None
        assert aexternal_ivr is not None
        assert IVRResult is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
