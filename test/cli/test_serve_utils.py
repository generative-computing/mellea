"""Unit tests for cli/serve/utils.py — finish_reason extraction."""

from unittest.mock import Mock

from mellea.core.base import ModelOutputThunk, RawProviderResponse
from mellea.serve.utils import extract_finish_reason


class TestExtractFinishReason:
    """Tests for extract_finish_reason function."""

    def test_default_finish_reason_when_no_raw(self):
        """Test that 'stop' is returned when output has no raw attribute."""
        output = Mock(spec=[])
        assert extract_finish_reason(output) == "stop"

    def test_default_finish_reason_when_raw_unset(self):
        """Test that 'stop' is returned when raw fields are unset."""
        output = ModelOutputThunk("test response")
        assert extract_finish_reason(output) == "stop"

    def test_ollama_done_reason_stop(self):
        """Test extraction of 'stop' from Ollama response.done_reason."""
        output = ModelOutputThunk("test response")
        chat_response = Mock()
        chat_response.done_reason = "stop"
        output.raw = RawProviderResponse(provider="ollama", response=chat_response)
        assert extract_finish_reason(output) == "stop"

    def test_ollama_done_reason_length(self):
        """Test extraction of 'length' from Ollama response.done_reason."""
        output = ModelOutputThunk("test response")
        chat_response = Mock()
        chat_response.done_reason = "length"
        output.raw = RawProviderResponse(provider="ollama", response=chat_response)
        assert extract_finish_reason(output) == "length"

    def test_ollama_done_reason_none(self):
        """Test that default 'stop' is returned when done_reason is None."""
        output = ModelOutputThunk("test response")
        chat_response = Mock()
        chat_response.done_reason = None
        output.raw = RawProviderResponse(provider="ollama", response=chat_response)
        assert extract_finish_reason(output) == "stop"

    def test_ollama_response_without_done_reason(self):
        """Test that default 'stop' is returned when response lacks done_reason."""
        output = ModelOutputThunk("test response")
        chat_response = Mock(spec=[])
        output.raw = RawProviderResponse(provider="ollama", response=chat_response)
        assert extract_finish_reason(output) == "stop"

    def test_openai_finish_reason_stop(self):
        """Test extraction of 'stop' from OpenAI response."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="openai",
            response={"choices": [{"finish_reason": "stop", "index": 0}]},
        )
        assert extract_finish_reason(output) == "stop"

    def test_openai_finish_reason_length(self):
        """Test extraction of 'length' from OpenAI response."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="openai",
            response={"choices": [{"finish_reason": "length", "index": 0}]},
        )
        assert extract_finish_reason(output) == "length"

    def test_openai_finish_reason_content_filter(self):
        """Test extraction of 'content_filter' from OpenAI response."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="openai",
            response={"choices": [{"finish_reason": "content_filter", "index": 0}]},
        )
        assert extract_finish_reason(output) == "content_filter"

    def test_openai_finish_reason_tool_calls(self):
        """Test extraction of 'tool_calls' from OpenAI response."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="openai",
            response={"choices": [{"finish_reason": "tool_calls", "index": 0}]},
        )
        assert extract_finish_reason(output) == "tool_calls"

    def test_openai_finish_reason_function_call(self):
        """Test extraction of 'function_call' from OpenAI response."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="openai",
            response={"choices": [{"finish_reason": "function_call", "index": 0}]},
        )
        assert extract_finish_reason(output) == "function_call"

    def test_openai_empty_choices_array(self):
        """Test that default 'stop' is returned when choices array is empty."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(provider="openai", response={"choices": []})
        assert extract_finish_reason(output) == "stop"

    def test_openai_missing_choices_key(self):
        """Test that default 'stop' is returned when choices key is missing."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(provider="openai", response={})
        assert extract_finish_reason(output) == "stop"

    def test_openai_finish_reason_none(self):
        """Test that default 'stop' is returned when finish_reason is None."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="openai",
            response={"choices": [{"finish_reason": None, "index": 0}]},
        )
        assert extract_finish_reason(output) == "stop"

    def test_openai_non_dict_response(self):
        """Test that default 'stop' is returned when response is not a dict."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(provider="openai", response="not a dict")
        assert extract_finish_reason(output) == "stop"

    def test_multiple_choices_uses_first(self):
        """Test that first choice is used when multiple choices exist."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="openai",
            response={
                "choices": [
                    {"finish_reason": "stop", "index": 0},
                    {"finish_reason": "length", "index": 1},
                ]
            },
        )
        assert extract_finish_reason(output) == "stop"

    def test_litellm_finish_reason_top_level_choices(self):
        """Test extraction of 'stop' from a LiteLLM top-level response with choices."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="litellm",
            response={"choices": [{"finish_reason": "stop", "index": 0}]},
        )
        assert extract_finish_reason(output) == "stop"

    def test_litellm_finish_reason_length(self):
        """Test extraction of 'length' from LiteLLM response."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="litellm",
            response={"choices": [{"finish_reason": "length", "index": 0}]},
        )
        assert extract_finish_reason(output) == "length"

    def test_litellm_finish_reason_tool_calls(self):
        """Test extraction of 'tool_calls' from LiteLLM response."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="litellm",
            response={"choices": [{"finish_reason": "tool_calls", "index": 0}]},
        )
        assert extract_finish_reason(output) == "tool_calls"

    def test_litellm_finish_reason_content_filter(self):
        """Test extraction of 'content_filter' from LiteLLM response."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="litellm",
            response={"choices": [{"finish_reason": "content_filter", "index": 0}]},
        )
        assert extract_finish_reason(output) == "content_filter"

    def test_litellm_finish_reason_function_call(self):
        """Test extraction of 'function_call' from LiteLLM response."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="litellm",
            response={"choices": [{"finish_reason": "function_call", "index": 0}]},
        )
        assert extract_finish_reason(output) == "function_call"

    def test_litellm_finish_reason_none(self):
        """Test that default 'stop' is returned when LiteLLM finish_reason is None."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="litellm",
            response={"choices": [{"finish_reason": None, "index": 0}]},
        )
        assert extract_finish_reason(output) == "stop"

    def test_litellm_missing_finish_reason_key(self):
        """Test that default 'stop' is returned when finish_reason key is missing."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(provider="litellm", response={})
        assert extract_finish_reason(output) == "stop"

    def test_litellm_non_dict_response(self):
        """Test that default 'stop' is returned when response is not a dict."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(provider="litellm", response="not a dict")
        assert extract_finish_reason(output) == "stop"

    def test_litellm_per_choice_dict_fallback(self):
        """Test that a LiteLLM per-choice dict (no choices key) uses top-level finish_reason."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(
            provider="litellm", response={"finish_reason": "tool_calls"}
        )
        assert extract_finish_reason(output) == "tool_calls"

    def test_huggingface_falls_through_to_default(self):
        """Test that Hugging Face (no finish_reason on response) returns default 'stop'."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(provider="huggingface", response=Mock(spec=[]))
        assert extract_finish_reason(output) == "stop"

    def test_unknown_provider_falls_through(self):
        """Test that an unknown provider returns default 'stop'."""
        output = ModelOutputThunk("test response")
        output.raw = RawProviderResponse(provider="experimental", response={})
        assert extract_finish_reason(output) == "stop"

    def test_tool_calls_attribute_short_circuits(self):
        """Test that a set tool_calls attribute returns 'tool_calls' regardless of raw."""
        output = ModelOutputThunk("test response", tool_calls={"fn": None})
        output.raw = RawProviderResponse(
            provider="openai",
            response={"choices": [{"finish_reason": "stop", "index": 0}]},
        )
        assert extract_finish_reason(output) == "tool_calls"
