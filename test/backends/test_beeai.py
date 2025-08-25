"""Test BeeAI backend integration."""

import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List

from mellea import MelleaSession
from mellea.stdlib.base import CBlock, LinearContext, ModelOutputThunk
from mellea.backends.beeai import BeeAIBackend
from mellea.backends.formatter import TemplateFormatter
from mellea.backends.types import ModelOption


class TestBeeAIBackend:
    """Test suite for BeeAI backend integration."""

    @pytest.fixture(scope="function")
    def backend(self):
        """Fresh BeeAI backend for each test."""
        return BeeAIBackend(
            model_id="granite3.3:2b",
            formatter=TemplateFormatter(model_id="granite3.3:2b"),
            base_url="http://localhost:11434"
        )

    @pytest.fixture(scope="function")
    def session(self, backend):
        """Fresh BeeAI session for each test."""
        session = MelleaSession(backend, ctx=LinearContext(is_chat_context=True))
        yield session
        session.reset()

    def test_backend_initialization(self, backend):
        """Test that BeeAI backend initializes correctly."""
        assert backend.model_id == "granite3.3:2b"
        assert backend.provider == "ollama"
        assert backend.base_url == "http://localhost:11434"
        assert isinstance(backend.formatter, TemplateFormatter)

    def test_backend_has_required_methods(self, backend):
        """Test that BeeAI backend implements required abstract methods."""
        assert hasattr(backend, 'generate_from_context')
        assert hasattr(backend, '_generate_from_raw')
        assert callable(backend.generate_from_context)
        assert callable(backend._generate_from_raw)

    @patch('mellea.backends.beeai.ChatModel')
    def test_generate_from_context_basic(self, mock_chat_model_class, session):
        """Test basic text generation from context."""
        # Mock the BeeAI chat model response
        mock_response = Mock()
        mock_response.messages = [Mock(content="The answer is 2.")]
        mock_response.usage = {"total_tokens": 10}
        
        mock_chat_model = Mock()
        mock_chat_model.create.return_value = mock_response
        mock_chat_model_class.from_name.return_value = mock_chat_model
        
        result = session.instruct("Compute 1+1.")
        
        assert isinstance(result, ModelOutputThunk)
        assert "2" in result.value
        assert result.is_computed()

    @patch('mellea.backends.beeai.ChatModel')
    def test_generate_from_context_with_format(self, mock_chat_model_class, session):
        """Test generation with structured output format."""
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
            age: int
        
        # Mock structured response
        mock_response = Mock()
        mock_response.messages = [Mock(content='{"name": "Alice", "age": 30}')]
        mock_response.usage = {"total_tokens": 15}
        
        mock_chat_model = Mock()
        mock_chat_model.create.return_value = mock_response
        mock_chat_model_class.from_name.return_value = mock_chat_model
        
        result = session.instruct(
            "Create a person named Alice who is 30 years old.",
            format=Person
        )
        
        assert isinstance(result, ModelOutputThunk)
        assert result.is_computed()
        
        # Parse the JSON response
        person = Person.model_validate_json(result.value)
        assert person.name == "Alice"
        assert person.age == 30

    @patch('mellea.backends.beeai.ChatModel')
    def test_generate_from_context_with_model_options(self, mock_chat_model_class, session):
        """Test generation with custom model options."""
        mock_response = Mock()
        mock_response.messages = [Mock(content="Response with custom options.")]
        mock_response.usage = {"total_tokens": 20}
        
        mock_chat_model = Mock()
        mock_chat_model.create.return_value = mock_response
        mock_chat_model_class.from_name.return_value = mock_chat_model
        
        result = session.instruct(
            "Generate a response.",
            model_options={ModelOption.TEMPERATURE: 0.7, ModelOption.MAX_NEW_TOKENS: 100}
        )
        
        assert isinstance(result, ModelOutputThunk)
        assert "custom options" in result.value.lower()

    @patch('mellea.backends.beeai.ChatModel')
    def test_generate_from_context_with_tool_calls(self, mock_chat_model_class, session):
        """Test generation with tool calls enabled."""
        mock_response = Mock()
        mock_response.messages = [Mock(content="I'll use the calculator tool to compute 2+2.")]
        mock_response.usage = {"total_tokens": 25}
        
        mock_chat_model = Mock()
        mock_chat_model.create.return_value = mock_response
        mock_chat_model_class.from_name.return_value = mock_chat_model
        
        # Create a component with template representation for tool calls
        from mellea.stdlib.base import TemplateRepresentation, Component
        
        class TestComponent(Component):
            def parts(self):
                return []
            
            def format_for_llm(self):
                return TemplateRepresentation(
                    obj=self,
                    args={"content": "Calculate 2+2"},
                    tools={"calculator": lambda x: eval(x)},
                    template_order=["*", "ContentBlock"]
                )
        
        action = TestComponent()
        
        result = session.backend.generate_from_context(
            action=action,
            ctx=session.ctx,
            tool_calls=True
        )
        
        assert isinstance(result, ModelOutputThunk)
        assert result.is_computed()

    @patch('mellea.backends.beeai.ChatModel')
    def test_generate_from_raw_batch(self, mock_chat_model_class, backend):
        """Test batch generation from raw inputs."""
        mock_responses = [
            Mock(messages=[Mock(content="Response 1")], usage={"total_tokens": 10}),
            Mock(messages=[Mock(content="Response 2")], usage={"total_tokens": 10}),
            Mock(messages=[Mock(content="Response 3")], usage={"total_tokens": 10})
        ]
        
        mock_chat_model = Mock()
        mock_chat_model.create.side_effect = mock_responses
        mock_chat_model_class.from_name.return_value = mock_chat_model
        
        actions = [
            CBlock(value="Prompt 1"),
            CBlock(value="Prompt 2"),
            CBlock(value="Prompt 3")
        ]
        
        results = backend._generate_from_raw(actions)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, ModelOutputThunk)
            assert f"Response {i+1}" in result.value

    @patch('mellea.backends.beeai.ChatModel')
    def test_generate_from_raw_with_format(self, mock_chat_model_class, backend):
        """Test raw generation with structured output format."""
        from pydantic import BaseModel
        
        class SimpleResponse(BaseModel):
            answer: str
        
        mock_response = Mock()
        mock_response.messages = [Mock(content='{"answer": "42"}')]
        mock_response.usage = {"total_tokens": 15}
        
        mock_chat_model = Mock()
        mock_chat_model.create.return_value = mock_response
        mock_chat_model_class.from_name.return_value = mock_chat_model
        
        actions = [CBlock(value="What is the answer to life, the universe, and everything?")]
        
        results = backend._generate_from_raw(
            actions,
            format=SimpleResponse
        )
        
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ModelOutputThunk)
        assert result.is_computed()
        
        # Parse the JSON response
        response = SimpleResponse.model_validate_json(result.value)
        assert response.answer == "42"

    @patch('mellea.backends.beeai.ChatModel')
    def test_error_handling(self, mock_chat_model_class, session):
        """Test error handling in the backend."""
        mock_chat_model = Mock()
        mock_chat_model.create.side_effect = Exception("API Error")
        mock_chat_model_class.from_name.return_value = mock_chat_model
        
        with pytest.raises(Exception) as exc_info:
            session.instruct("This should fail.")
        
        assert "API Error" in str(exc_info.value)

    @patch('mellea.backends.beeai.ChatModel')
    def test_context_handling(self, mock_chat_model_class, session):
        """Test that context is properly handled across multiple turns."""
        mock_responses = [
            Mock(messages=[Mock(content="Paris is the capital of France.")], usage={"total_tokens": 10}),
            Mock(messages=[Mock(content="I previously said that Paris is the capital of France.")], usage={"total_tokens": 15})
        ]
        
        mock_chat_model = Mock()
        mock_chat_model.create.side_effect = mock_responses
        mock_chat_model_class.from_name.return_value = mock_chat_model
        
        # First turn
        session.instruct("What is the capital of France?")
        
        # Second turn - should have context from first turn
        result = session.instruct("What did I just ask you?")
        
        assert isinstance(result, ModelOutputThunk)
        assert "Paris" in result.value
        assert "capital of France" in result.value

    def test_model_options_merging(self, backend):
        """Test that model options are properly merged."""
        # Test with default options
        backend.model_options = {ModelOption.TEMPERATURE: 0.5}
        
        # Test merging with call-specific options
        merged_options = backend._merge_model_options({ModelOption.MAX_NEW_TOKENS: 100})
        
        assert merged_options[ModelOption.TEMPERATURE] == 0.5
        assert merged_options[ModelOption.MAX_NEW_TOKENS] == 100

    def test_backend_repr(self, backend):
        """Test backend string representation."""
        repr_str = repr(backend)
        assert "BeeAIBackend" in repr_str
        assert "granite3.3:2b" in repr_str
        assert "ollama" in repr_str

    @patch('mellea.backends.beeai.ChatModel')
    def test_generate_with_system_prompt(self, mock_chat_model_class, session):
        """Test generation with system prompt in model options."""
        mock_response = Mock()
        mock_response.messages = [Mock(content="I am a helpful assistant. The answer is 3.")]
        mock_response.usage = {"total_tokens": 20}
        
        mock_chat_model = Mock()
        mock_chat_model.create.return_value = mock_response
        mock_chat_model_class.from_name.return_value = mock_chat_model
        
        result = session.instruct(
            "What is 1+2?",
            model_options={ModelOption.SYSTEM_PROMPT: "You are a helpful math assistant."}
        )
        
        assert isinstance(result, ModelOutputThunk)
        assert "3" in result.value
        assert "helpful assistant" in result.value.lower()


if __name__ == "__main__":
    pytest.main([__file__])
