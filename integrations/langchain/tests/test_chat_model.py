"""Tests for MelleaChatModel."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from mellea_langchain import MelleaChatModel


class MockMelleaSession:
    """Mock Mellea session for testing."""

    def __init__(self, response_content="Test response"):
        self.response_content = response_content
        self.last_message = None
        self.last_model_options = None
        self.last_tool_calls = None

    def chat(self, message, model_options=None, tool_calls=False):
        """Mock sync chat method."""
        self.last_message = message
        self.last_model_options = model_options
        self.last_tool_calls = tool_calls

        class MockResponse:
            def __init__(self, content):
                self.content = content
                self._tool_calls = None

        return MockResponse(self.response_content)

    async def achat(self, message, model_options=None, tool_calls=False):
        """Mock async chat method."""
        self.last_message = message
        self.last_model_options = model_options
        self.last_tool_calls = tool_calls

        class MockResponse:
            def __init__(self, content):
                self.content = content
                self._tool_calls = None

        return MockResponse(self.response_content)


class TestMelleaChatModel:
    """Test MelleaChatModel functionality."""

    def test_initialization(self):
        """Test chat model initialization."""
        mock_session = MockMelleaSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        assert chat_model.mellea_session == mock_session
        assert chat_model.model_name == "mellea"
        assert chat_model._llm_type == "mellea"

    def test_initialization_with_custom_name(self):
        """Test initialization with custom model name."""
        mock_session = MockMelleaSession()
        chat_model = MelleaChatModel(mellea_session=mock_session, model_name="custom-mellea")

        assert chat_model.model_name == "custom-mellea"

    @pytest.mark.asyncio
    async def test_agenerate(self):
        """Test async generation."""
        mock_session = MockMelleaSession(response_content="Hello from Mellea!")
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content="Hello")]
        result = await chat_model._agenerate(messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Hello from Mellea!"
        assert mock_session.last_message == "Hello"

    @pytest.mark.asyncio
    async def test_agenerate_with_system_message(self):
        """Test async generation with system message."""
        mock_session = MockMelleaSession(response_content="Response")
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hi"),
        ]
        result = await chat_model._agenerate(messages)

        assert len(result.generations) == 1
        assert mock_session.last_message == "Hi"

    @pytest.mark.asyncio
    async def test_agenerate_with_model_options(self):
        """Test async generation with model options."""
        mock_session = MockMelleaSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content="Test")]
        model_options = {"temperature": 0.7, "max_tokens": 100}

        await chat_model._agenerate(messages, model_options=model_options)

        assert mock_session.last_model_options == model_options

    @pytest.mark.asyncio
    async def test_agenerate_empty_messages(self):
        """Test async generation with empty messages list."""
        mock_session = MockMelleaSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        with pytest.raises(ValueError, match="No messages provided"):
            await chat_model._agenerate([])

    @pytest.mark.asyncio
    async def test_agenerate_multiple_messages(self):
        """Test async generation with conversation history."""
        mock_session = MockMelleaSession(response_content="I remember!")
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [
            HumanMessage(content="My name is Alice"),
            HumanMessage(content="What's my name?"),
        ]
        result = await chat_model._agenerate(messages)

        # Should use the last message for generation
        assert mock_session.last_message == "What's my name?"
        assert result.generations[0].message.content == "I remember!"


class TestMelleaChatModelStreaming:
    """Test streaming functionality."""

    @pytest.mark.asyncio
    async def test_astream_basic(self):
        """Test basic async streaming."""

        class MockStreamingSession:
            async def achat(self, message, model_options=None, tool_calls=False):
                class MockResponse:
                    def __init__(self):
                        self.content = "Full response"
                        self._tool_calls = None

                return MockResponse()

        mock_session = MockStreamingSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content="Test")]
        chunks = []

        async for chunk in chat_model._astream(messages):
            chunks.append(chunk)

        # Should get at least one chunk
        assert len(chunks) >= 1


class TestMelleaChatModelToolBinding:
    """Test tool binding functionality."""

    def test_bind_tools_creates_new_instance(self):
        """Test that bind_tools creates a new model instance."""
        mock_session = MockMelleaSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        # Mock tool
        class MockTool:
            name = "test_tool"
            description = "A test tool"

        tools = [MockTool()]
        bound_model = chat_model.bind_tools(tools)

        # Should be a new instance
        assert bound_model is not chat_model
        assert isinstance(bound_model, MelleaChatModel)
        assert bound_model.mellea_session == mock_session

    def test_bind_tools_stores_tools(self):
        """Test that bind_tools stores tools on the model."""
        mock_session = MockMelleaSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        class MockTool:
            name = "search"

        tools = [MockTool()]
        bound_model = chat_model.bind_tools(tools)

        assert hasattr(bound_model, "_bound_tools")
        assert bound_model._bound_tools == tools

    def test_bind_tools_with_tool_choice(self):
        """Test bind_tools with tool_choice parameter."""
        mock_session = MockMelleaSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        class MockTool:
            name = "calculator"

        tools = [MockTool()]
        bound_model = chat_model.bind_tools(tools, tool_choice="auto")

        assert hasattr(bound_model, "_tool_choice")
        assert bound_model._tool_choice == "auto"

    @pytest.mark.asyncio
    async def test_agenerate_with_bound_tools(self):
        """Test async generation with bound tools."""

        class MockSessionWithTools:
            def __init__(self):
                self.last_tool_calls = None
                self.last_model_options = None

            async def achat(self, message, model_options=None, tool_calls=False):
                self.last_tool_calls = tool_calls
                self.last_model_options = model_options

                class MockResponse:
                    def __init__(self):
                        self.content = "Using tools"
                        self._tool_calls = None

                return MockResponse()

        mock_session = MockSessionWithTools()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        class MockTool:
            name = "web_search"

        bound_model = chat_model.bind_tools([MockTool()])

        messages = [HumanMessage(content="Search for Python")]
        result = await bound_model._agenerate(messages)

        # Should enable tool calls
        assert mock_session.last_tool_calls is True
        assert result.generations[0].message.content == "Using tools"

    def test_bind_tools_preserves_model_config(self):
        """Test that bind_tools preserves model configuration."""
        mock_session = MockMelleaSession()
        chat_model = MelleaChatModel(
            mellea_session=mock_session, model_name="custom-model", streaming=True
        )

        class MockTool:
            name = "tool"

        bound_model = chat_model.bind_tools([MockTool()])

        assert bound_model.model_name == "custom-model"
        assert bound_model.streaming is True


class TestMelleaChatModelSynchronous:
    """Test synchronous generation methods."""

    def test_generate_basic(self):
        """Test basic synchronous generation."""
        mock_session = MockMelleaSession(response_content="Sync response")
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content="Hello")]
        result = chat_model._generate(messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Sync response"

    def test_generate_with_model_options(self):
        """Test synchronous generation with model options."""
        mock_session = MockMelleaSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content="Test")]
        model_options = {"temperature": 0.5}

        chat_model._generate(messages, model_options=model_options)

        assert mock_session.last_model_options == model_options

    def test_generate_empty_messages(self):
        """Test synchronous generation with empty messages."""
        mock_session = MockMelleaSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        with pytest.raises(ValueError, match="No messages provided"):
            chat_model._generate([])

    def test_stream_basic(self):
        """Test basic synchronous streaming."""

        class MockStreamSession:
            def chat(self, message, model_options=None, tool_calls=False):
                class MockResponse:
                    def __init__(self):
                        self.content = "Streamed content"
                        self._tool_calls = None

                return MockResponse()

            async def achat(self, message, model_options=None, tool_calls=False):
                class MockResponse:
                    def __init__(self):
                        self.content = "Streamed content"
                        self._tool_calls = None

                return MockResponse()

        mock_session = MockStreamSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content="Stream test")]
        chunks = list(chat_model._stream(messages))

        assert len(chunks) >= 1
        # Should have content
        assert any(chunk.message.content for chunk in chunks)

    def test_stream_with_callback(self):
        """Test synchronous streaming with callback manager."""

        class MockStreamSession:
            def chat(self, message, model_options=None, tool_calls=False):
                class MockResponse:
                    def __init__(self):
                        self.content = "Content"
                        self._tool_calls = None

                return MockResponse()

            async def achat(self, message, model_options=None, tool_calls=False):
                class MockResponse:
                    def __init__(self):
                        self.content = "Content"
                        self._tool_calls = None

                return MockResponse()

        mock_session = MockStreamSession()
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content="Test")]
        chunks = list(chat_model._stream(messages))

        # Since streaming is not supported, should return single chunk
        assert len(chunks) == 1
        assert chunks[0].message.content == "Content"


class TestMelleaChatModelErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_agenerate_session_error(self):
        """Test handling of Mellea session errors."""

        class ErrorSession:
            async def achat(self, message, model_options=None, tool_calls=False):
                raise RuntimeError("Mellea API error")

        error_session = ErrorSession()
        chat_model = MelleaChatModel(mellea_session=error_session)

        messages = [HumanMessage(content="Test")]

        with pytest.raises(RuntimeError, match="Mellea API error"):
            await chat_model._agenerate(messages)

    @pytest.mark.asyncio
    async def test_agenerate_invalid_response(self):
        """Test handling of invalid Mellea response."""

        class InvalidResponseSession:
            async def achat(self, message, model_options=None, tool_calls=False):
                # Return object without content attribute
                class InvalidResponse:
                    pass

                return InvalidResponse()

        session = InvalidResponseSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]
        result = await chat_model._agenerate(messages)

        # Should handle gracefully with empty content
        assert result.generations[0].message.content == ""

    def test_generate_with_none_session(self):
        """Test initialization with None session."""
        # Should allow None session (will fail at generation time)
        chat_model = MelleaChatModel(mellea_session=None)
        assert chat_model.mellea_session is None

    @pytest.mark.asyncio
    async def test_agenerate_with_unicode_content(self):
        """Test generation with unicode characters."""
        mock_session = MockMelleaSession(response_content="Hello 世界 🌍")
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content="Unicode test: 你好")]
        result = await chat_model._agenerate(messages)

        assert result.generations[0].message.content == "Hello 世界 🌍"
        assert mock_session.last_message == "Unicode test: 你好"

    @pytest.mark.asyncio
    async def test_agenerate_with_very_long_message(self):
        """Test generation with very long message."""
        long_content = "A" * 10000
        mock_session = MockMelleaSession(response_content="Response")
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content=long_content)]
        result = await chat_model._agenerate(messages)

        assert mock_session.last_message == long_content
        assert result.generations[0].message.content == "Response"

    @pytest.mark.asyncio
    async def test_agenerate_with_empty_content(self):
        """Test generation with empty message content."""
        mock_session = MockMelleaSession(response_content="Response")
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content="")]
        result = await chat_model._agenerate(messages)

        assert mock_session.last_message == ""
        assert result.generations[0].message.content == "Response"

    @pytest.mark.asyncio
    async def test_agenerate_with_special_characters(self):
        """Test generation with special characters."""
        special_content = "Test\n\t<>&\"'`"
        mock_session = MockMelleaSession(response_content="OK")
        chat_model = MelleaChatModel(mellea_session=mock_session)

        messages = [HumanMessage(content=special_content)]
        await chat_model._agenerate(messages)

        assert mock_session.last_message == special_content


class TestMelleaChatModelStreamingAdvanced:
    """Test advanced streaming scenarios."""

    @pytest.mark.asyncio
    async def test_astream_with_queue(self):
        """Test async streaming (returns full response as single chunk)."""

        class QueueStreamSession:
            async def achat(self, message, model_options=None, tool_calls=False):
                class StreamResponse:
                    def __init__(self):
                        self.content = "Full response"
                        self._tool_calls = None

                return StreamResponse()

        session = QueueStreamSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]
        chunks = []

        async for chunk in chat_model._astream(messages):
            chunks.append(chunk.message.content)

        # Since streaming is not supported, returns single chunk with full response
        assert len(chunks) == 1
        assert chunks[0] == "Full response"

    @pytest.mark.asyncio
    async def test_astream_without_queue_fallback(self):
        """Test async streaming fallback when no queue available."""

        class NoQueueSession:
            async def achat(self, message, model_options=None, tool_calls=False):
                class SimpleResponse:
                    def __init__(self):
                        self.content = "No streaming support"
                        self._tool_calls = None

                return SimpleResponse()

        session = NoQueueSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]
        chunks = []

        async for chunk in chat_model._astream(messages):
            chunks.append(chunk)

        # Should get single chunk with full content
        assert len(chunks) == 1
        assert chunks[0].message.content == "No streaming support"

    @pytest.mark.asyncio
    async def test_astream_with_callback(self):
        """Test async streaming."""

        class CallbackStreamSession:
            async def achat(self, message, model_options=None, tool_calls=False):
                class StreamResponse:
                    def __init__(self):
                        self.content = "Response"
                        self._tool_calls = None

                return StreamResponse()

        session = CallbackStreamSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]
        chunks = []

        async for chunk in chat_model._astream(messages):
            chunks.append(chunk)

        # Since streaming is not supported, returns single chunk
        assert len(chunks) == 1
        assert chunks[0].message.content == "Response"

    @pytest.mark.asyncio
    async def test_astream_empty_chunks(self):
        """Test async streaming with empty content."""

        class EmptyChunkSession:
            async def achat(self, message, model_options=None, tool_calls=False):
                class StreamResponse:
                    def __init__(self):
                        self.content = ""
                        self._tool_calls = None

                return StreamResponse()

        session = EmptyChunkSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]
        chunks = []

        async for chunk in chat_model._astream(messages):
            chunks.append(chunk.message.content)

        # Since streaming is not supported, returns single chunk (even if empty)
        assert len(chunks) == 1
        assert chunks[0] == ""

    @pytest.mark.asyncio
    async def test_astream_with_model_options(self):
        """Test async streaming with model options."""

        class OptionsStreamSession:
            def __init__(self):
                self.last_model_options = None

            async def achat(self, message, model_options=None, tool_calls=False):
                self.last_model_options = model_options

                class StreamResponse:
                    def __init__(self):
                        self.content = "Response"
                        self._tool_calls = None

                return StreamResponse()

        session = OptionsStreamSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]
        model_options = {"temperature": 0.8}

        chunks = []
        async for chunk in chat_model._astream(messages, model_options=model_options):
            chunks.append(chunk)

        # Should pass model options and enable streaming
        assert session.last_model_options is not None
        assert "temperature" in session.last_model_options
        assert session.last_model_options["temperature"] == 0.8


class TestMelleaChatModelRequirementsAndStrategy:
    """Test requirements and strategy functionality."""

    @pytest.mark.asyncio
    async def test_agenerate_with_requirements(self):
        """Test async generation with requirements."""

        class MockInstructSession:
            def __init__(self):
                self.last_requirements = None
                self.last_strategy = None
                self.last_return_sampling_results = None

            async def ainstruct(
                self,
                message,
                requirements=None,
                strategy=None,
                model_options=None,
                return_sampling_results=False,
            ):
                self.last_requirements = requirements
                self.last_strategy = strategy
                self.last_return_sampling_results = return_sampling_results

                class MockResponse:
                    def __init__(self):
                        self.content = "Response with requirements"
                        self._tool_calls = None

                return MockResponse()

            async def achat(self, message, model_options=None, tool_calls=False):
                # Fallback for non-requirement calls
                class MockResponse:
                    def __init__(self):
                        self.content = "Regular response"
                        self._tool_calls = None

                return MockResponse()

        session = MockInstructSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]
        requirements = ["Requirement 1", "Requirement 2"]

        result = await chat_model._agenerate(messages, model_options={"requirements": requirements})

        assert session.last_requirements == requirements
        assert result.generations[0].message.content == "Response with requirements"

    @pytest.mark.asyncio
    async def test_agenerate_with_strategy(self):
        """Test async generation with sampling strategy."""

        class MockStrategy:
            def __init__(self):
                self.loop_budget = 5

        class MockInstructSession:
            def __init__(self):
                self.last_strategy = None

            async def ainstruct(
                self,
                message,
                requirements=None,
                strategy=None,
                model_options=None,
                return_sampling_results=False,
            ):
                self.last_strategy = strategy

                class MockResponse:
                    def __init__(self):
                        self.content = "Response with strategy"
                        self._tool_calls = None

                return MockResponse()

            async def achat(self, message, model_options=None, tool_calls=False):
                class MockResponse:
                    def __init__(self):
                        self.content = "Regular response"
                        self._tool_calls = None

                return MockResponse()

        session = MockInstructSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]
        strategy = MockStrategy()

        result = await chat_model._agenerate(messages, model_options={"strategy": strategy})

        assert session.last_strategy == strategy
        assert result.generations[0].message.content == "Response with strategy"

    @pytest.mark.asyncio
    async def test_agenerate_with_sampling_results_success(self):
        """Test async generation with return_sampling_results=True (success case)."""

        class MockSamplingResult:
            def __init__(self):
                self.success = True
                self.result = type("obj", (), {"content": "Successful result"})()
                self.sample_generations = []

        class MockInstructSession:
            async def ainstruct(
                self,
                message,
                requirements=None,
                strategy=None,
                model_options=None,
                return_sampling_results=False,
            ):
                return MockSamplingResult()

            async def achat(self, message, model_options=None, tool_calls=False):
                class MockResponse:
                    def __init__(self):
                        self.content = "Regular response"
                        self._tool_calls = None

                return MockResponse()

        session = MockInstructSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]

        result = await chat_model._agenerate(
            messages,
            model_options={
                "requirements": ["Test requirement"],
                "return_sampling_results": True,
            },
        )

        assert result.generations[0].message.content == "Successful result"

    @pytest.mark.asyncio
    async def test_agenerate_with_sampling_results_failure(self):
        """Test async generation with return_sampling_results=True (failure case)."""

        class MockSamplingResult:
            def __init__(self):
                self.success = False
                self.result = None
                self.sample_generations = [
                    type(
                        "obj",
                        (),
                        {"content": "Fallback result", "value": "Fallback result"},
                    )()
                ]

        class MockInstructSession:
            async def ainstruct(
                self,
                message,
                requirements=None,
                strategy=None,
                model_options=None,
                return_sampling_results=False,
            ):
                return MockSamplingResult()

            async def achat(self, message, model_options=None, tool_calls=False):
                class MockResponse:
                    def __init__(self):
                        self.content = "Regular response"
                        self._tool_calls = None

                return MockResponse()

        session = MockInstructSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]

        result = await chat_model._agenerate(
            messages,
            model_options={
                "requirements": ["Test requirement"],
                "strategy": type("obj", (), {})(),
                "return_sampling_results": True,
            },
        )

        assert result.generations[0].message.content == "Fallback result"

    @pytest.mark.asyncio
    async def test_agenerate_with_requirements_and_strategy(self):
        """Test async generation with both requirements and strategy."""

        class MockInstructSession:
            def __init__(self):
                self.last_requirements = None
                self.last_strategy = None

            async def ainstruct(
                self,
                message,
                requirements=None,
                strategy=None,
                model_options=None,
                return_sampling_results=False,
            ):
                self.last_requirements = requirements
                self.last_strategy = strategy

                class MockResponse:
                    def __init__(self):
                        self.content = "Response with both"
                        self._tool_calls = None

                return MockResponse()

            async def achat(self, message, model_options=None, tool_calls=False):
                class MockResponse:
                    def __init__(self):
                        self.content = "Regular response"
                        self._tool_calls = None

                return MockResponse()

        session = MockInstructSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]
        requirements = ["Req 1", "Req 2"]
        strategy = type("MockStrategy", (), {"loop_budget": 3})()

        result = await chat_model._agenerate(
            messages, model_options={"requirements": requirements, "strategy": strategy}
        )

        assert session.last_requirements == requirements
        assert session.last_strategy == strategy
        assert result.generations[0].message.content == "Response with both"

    @pytest.mark.asyncio
    async def test_agenerate_without_requirements_uses_achat(self):
        """Test that generation without requirements uses achat method."""

        class MockSession:
            def __init__(self):
                self.achat_called = False
                self.ainstruct_called = False

            async def achat(self, message, model_options=None, tool_calls=False):
                self.achat_called = True

                class MockResponse:
                    def __init__(self):
                        self.content = "Chat response"
                        self._tool_calls = None

                return MockResponse()

            async def ainstruct(
                self,
                message,
                requirements=None,
                strategy=None,
                model_options=None,
                return_sampling_results=False,
            ):
                self.ainstruct_called = True

                class MockResponse:
                    def __init__(self):
                        self.content = "Instruct response"
                        self._tool_calls = None

                return MockResponse()

        session = MockSession()
        chat_model = MelleaChatModel(mellea_session=session)

        messages = [HumanMessage(content="Test")]
        result = await chat_model._agenerate(messages)

        assert session.achat_called is True
        assert session.ainstruct_called is False
        assert result.generations[0].message.content == "Chat response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
