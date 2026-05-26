"""Unit tests for _build_model_options function."""

from cli.serve.app import _build_model_options
from cli.serve.models import ChatCompletionRequest, ChatMessage
from mellea.backends.model_options import ModelOption


class TestBuildModelOptions:
    """Direct unit tests for _build_model_options."""

    def test_temperature_mapping(self):
        """Test that temperature is correctly mapped to ModelOption.TEMPERATURE."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            temperature=0.7,
        )
        options = _build_model_options(request)
        assert options[ModelOption.TEMPERATURE] == 0.7

    def test_max_tokens_mapping(self):
        """Test that max_tokens is correctly mapped to ModelOption.MAX_NEW_TOKENS."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            max_tokens=100,
        )
        options = _build_model_options(request)
        assert options[ModelOption.MAX_NEW_TOKENS] == 100

    def test_seed_mapping(self):
        """Test that seed is correctly mapped to ModelOption.SEED."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            seed=42,
        )
        options = _build_model_options(request)
        assert options[ModelOption.SEED] == 42

    def test_multiple_options(self):
        """Test that multiple options are correctly mapped together."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            temperature=0.8,
            max_tokens=200,
            seed=123,
        )
        options = _build_model_options(request)
        assert options[ModelOption.TEMPERATURE] == 0.8
        assert options[ModelOption.MAX_NEW_TOKENS] == 200
        assert options[ModelOption.SEED] == 123

    def test_excluded_fields_not_in_output(self):
        """Test that excluded fields are not included in model_options."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            n=1,
            user="test-user",
            stream=False,
            temperature=0.5,
        )
        options = _build_model_options(request)
        # Check that excluded fields are not present
        assert "model" not in options
        assert "messages" not in options
        assert "n" not in options
        assert "user" not in options
        assert "stream" not in options
        assert ModelOption.STREAM not in options
        # Check that temperature is present
        assert ModelOption.TEMPERATURE in options

    def test_none_values_excluded(self):
        """Test that None values are excluded from output."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            temperature=None,
            max_tokens=None,
        )
        options = _build_model_options(request)
        assert ModelOption.TEMPERATURE not in options
        assert ModelOption.MAX_NEW_TOKENS not in options

    def test_minimal_request_includes_defaults(self):
        """Test that a minimal request includes default values like temperature."""
        request = ChatCompletionRequest(
            model="test-model", messages=[ChatMessage(role="user", content="test")]
        )
        options = _build_model_options(request)
        # ChatCompletionRequest has default temperature=1.0
        # stream is excluded from model_options (handled separately in endpoint logic)
        assert options == {ModelOption.TEMPERATURE: 1.0}

    def test_requirements_excluded(self):
        """Test that requirements field is excluded from model_options."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            requirements=["req1", "req2"],
            temperature=0.7,
        )
        options = _build_model_options(request)
        assert "requirements" not in options
        assert ModelOption.TEMPERATURE in options
