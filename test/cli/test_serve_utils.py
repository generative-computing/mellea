"""Unit tests for cli/serve/utils.py — finish_reason extraction."""

from unittest.mock import Mock

import pytest

from cli.serve.utils import extract_finish_reason
from mellea.core.base import ModelOutputThunk


class TestExtractFinishReason:
    """Tests for extract_finish_reason function."""

    def test_default_finish_reason_when_no_meta(self):
        """Test that 'stop' is returned when output has no _meta attribute."""
        output = ModelOutputThunk("test response")
        # Don't set _meta attribute
        assert extract_finish_reason(output) == "stop"

    def test_default_finish_reason_when_meta_is_none(self):
        """Test that 'stop' is returned when _meta is None."""
        output = ModelOutputThunk("test response")
        output._meta = None
        assert extract_finish_reason(output) == "stop"

    def test_default_finish_reason_when_meta_is_empty(self):
        """Test that 'stop' is returned when _meta is empty dict."""
        output = ModelOutputThunk("test response")
        output._meta = {}
        assert extract_finish_reason(output) == "stop"

    def test_ollama_done_reason_stop(self):
        """Test extraction of 'stop' from Ollama chat_response.done_reason."""
        output = ModelOutputThunk("test response")
        chat_response = Mock()
        chat_response.done_reason = "stop"
        output._meta = {"chat_response": chat_response}
        assert extract_finish_reason(output) == "stop"

    def test_ollama_done_reason_length(self):
        """Test extraction of 'length' from Ollama chat_response.done_reason."""
        output = ModelOutputThunk("test response")
        chat_response = Mock()
        chat_response.done_reason = "length"
        output._meta = {"chat_response": chat_response}
        assert extract_finish_reason(output) == "length"

    def test_ollama_done_reason_none(self):
        """Test that default 'stop' is returned when done_reason is None."""
        output = ModelOutputThunk("test response")
        chat_response = Mock()
        chat_response.done_reason = None
        output._meta = {"chat_response": chat_response}
        assert extract_finish_reason(output) == "stop"

    def test_ollama_chat_response_without_done_reason(self):
        """Test that default 'stop' is returned when chat_response lacks done_reason."""
        output = ModelOutputThunk("test response")
        chat_response = Mock(spec=[])  # Mock without done_reason attribute
        output._meta = {"chat_response": chat_response}
        assert extract_finish_reason(output) == "stop"

    def test_openai_finish_reason_stop(self):
        """Test extraction of 'stop' from OpenAI oai_chat_response."""
        output = ModelOutputThunk("test response")
        output._meta = {
            "oai_chat_response": {"choices": [{"finish_reason": "stop", "index": 0}]}
        }
        assert extract_finish_reason(output) == "stop"

    def test_openai_finish_reason_length(self):
        """Test extraction of 'length' from OpenAI oai_chat_response."""
        output = ModelOutputThunk("test response")
        output._meta = {
            "oai_chat_response": {"choices": [{"finish_reason": "length", "index": 0}]}
        }
        assert extract_finish_reason(output) == "length"

    def test_openai_finish_reason_content_filter(self):
        """Test extraction of 'content_filter' from OpenAI oai_chat_response."""
        output = ModelOutputThunk("test response")
        output._meta = {
            "oai_chat_response": {
                "choices": [{"finish_reason": "content_filter", "index": 0}]
            }
        }
        assert extract_finish_reason(output) == "content_filter"

    def test_openai_finish_reason_tool_calls(self):
        """Test extraction of 'tool_calls' from OpenAI oai_chat_response."""
        output = ModelOutputThunk("test response")
        output._meta = {
            "oai_chat_response": {
                "choices": [{"finish_reason": "tool_calls", "index": 0}]
            }
        }
        assert extract_finish_reason(output) == "tool_calls"

    def test_openai_finish_reason_function_call(self):
        """Test extraction of 'function_call' from OpenAI oai_chat_response."""
        output = ModelOutputThunk("test response")
        output._meta = {
            "oai_chat_response": {
                "choices": [{"finish_reason": "function_call", "index": 0}]
            }
        }
        assert extract_finish_reason(output) == "function_call"

    def test_openai_empty_choices_array(self):
        """Test that default 'stop' is returned when choices array is empty."""
        output = ModelOutputThunk("test response")
        output._meta = {"oai_chat_response": {"choices": []}}
        assert extract_finish_reason(output) == "stop"

    def test_openai_missing_choices_key(self):
        """Test that default 'stop' is returned when choices key is missing."""
        output = ModelOutputThunk("test response")
        output._meta = {"oai_chat_response": {}}
        assert extract_finish_reason(output) == "stop"

    def test_openai_finish_reason_none(self):
        """Test that default 'stop' is returned when finish_reason is None."""
        output = ModelOutputThunk("test response")
        output._meta = {
            "oai_chat_response": {"choices": [{"finish_reason": None, "index": 0}]}
        }
        assert extract_finish_reason(output) == "stop"

    def test_openai_non_dict_response(self):
        """Test that default 'stop' is returned when oai_chat_response is not a dict."""
        output = ModelOutputThunk("test response")
        output._meta = {"oai_chat_response": "not a dict"}
        assert extract_finish_reason(output) == "stop"

    def test_ollama_takes_precedence_over_openai(self):
        """Test that Ollama done_reason is checked before OpenAI finish_reason."""
        output = ModelOutputThunk("test response")
        chat_response = Mock()
        chat_response.done_reason = "length"
        output._meta = {
            "chat_response": chat_response,
            "oai_chat_response": {"choices": [{"finish_reason": "stop", "index": 0}]},
        }
        # Should return Ollama's done_reason, not OpenAI's finish_reason
        assert extract_finish_reason(output) == "length"

    def test_openai_used_when_ollama_missing(self):
        """Test that OpenAI finish_reason is used when Ollama data is missing."""
        output = ModelOutputThunk("test response")
        output._meta = {
            "oai_chat_response": {
                "choices": [{"finish_reason": "content_filter", "index": 0}]
            }
        }
        assert extract_finish_reason(output) == "content_filter"

    def test_multiple_choices_uses_first(self):
        """Test that first choice is used when multiple choices exist."""
        output = ModelOutputThunk("test response")
        output._meta = {
            "oai_chat_response": {
                "choices": [
                    {"finish_reason": "stop", "index": 0},
                    {"finish_reason": "length", "index": 1},
                ]
            }
        }
        assert extract_finish_reason(output) == "stop"

    def test_other_meta_keys_ignored(self):
        """Test that unrelated _meta keys don't interfere."""
        output = ModelOutputThunk("test response")
        output._meta = {
            "model": "gpt-4",
            "provider": "openai",
            "usage": {"total_tokens": 100},
            "random_key": "random_value",
        }
        assert extract_finish_reason(output) == "stop"

    def test_output_without_meta_attribute(self):
        """Test handling of output objects that don't have _meta attribute at all."""
        # Create a simple object without _meta
        output = Mock(spec=[])
        assert extract_finish_reason(output) == "stop"

    def test_litellm_finish_reason_stop(self):
        """Test extraction of 'stop' from LiteLLM litellm_chat_response."""
        output = ModelOutputThunk("test response")
        output._meta = {"litellm_chat_response": {"finish_reason": "stop"}}
        assert extract_finish_reason(output) == "stop"

    def test_litellm_finish_reason_length(self):
        """Test extraction of 'length' from LiteLLM litellm_chat_response."""
        output = ModelOutputThunk("test response")
        output._meta = {"litellm_chat_response": {"finish_reason": "length"}}
        assert extract_finish_reason(output) == "length"

    def test_litellm_finish_reason_tool_calls(self):
        """Test extraction of 'tool_calls' from LiteLLM litellm_chat_response."""
        output = ModelOutputThunk("test response")
        output._meta = {"litellm_chat_response": {"finish_reason": "tool_calls"}}
        assert extract_finish_reason(output) == "tool_calls"

    def test_litellm_finish_reason_content_filter(self):
        """Test extraction of 'content_filter' from LiteLLM litellm_chat_response."""
        output = ModelOutputThunk("test response")
        output._meta = {"litellm_chat_response": {"finish_reason": "content_filter"}}
        assert extract_finish_reason(output) == "content_filter"

    def test_litellm_finish_reason_function_call(self):
        """Test extraction of 'function_call' from LiteLLM litellm_chat_response."""
        output = ModelOutputThunk("test response")
        output._meta = {"litellm_chat_response": {"finish_reason": "function_call"}}
        assert extract_finish_reason(output) == "function_call"

    def test_litellm_finish_reason_none(self):
        """Test that default 'stop' is returned when LiteLLM finish_reason is None."""
        output = ModelOutputThunk("test response")
        output._meta = {"litellm_chat_response": {"finish_reason": None}}
        assert extract_finish_reason(output) == "stop"

    def test_litellm_missing_finish_reason_key(self):
        """Test that default 'stop' is returned when finish_reason key is missing."""
        output = ModelOutputThunk("test response")
        output._meta = {"litellm_chat_response": {}}
        assert extract_finish_reason(output) == "stop"

    def test_litellm_non_dict_response(self):
        """Test that default 'stop' is returned when litellm_chat_response is not a dict."""
        output = ModelOutputThunk("test response")
        output._meta = {"litellm_chat_response": "not a dict"}
        assert extract_finish_reason(output) == "stop"

    def test_backend_precedence_ollama_openai_litellm(self):
        """Test that backends are checked in order: Ollama, OpenAI, LiteLLM."""
        output = ModelOutputThunk("test response")
        chat_response = Mock()
        chat_response.done_reason = "length"
        output._meta = {
            "chat_response": chat_response,
            "oai_chat_response": {"choices": [{"finish_reason": "stop", "index": 0}]},
            "litellm_chat_response": {"finish_reason": "content_filter"},
        }
        # Should return Ollama's done_reason (checked first)
        assert extract_finish_reason(output) == "length"

    def test_litellm_used_when_ollama_and_openai_missing(self):
        """Test that LiteLLM finish_reason is used when Ollama and OpenAI data missing."""
        output = ModelOutputThunk("test response")
        output._meta = {"litellm_chat_response": {"finish_reason": "tool_calls"}}
        assert extract_finish_reason(output) == "tool_calls"

    def test_openai_takes_precedence_over_litellm(self):
        """Test that OpenAI finish_reason is checked before LiteLLM."""
        output = ModelOutputThunk("test response")
        output._meta = {
            "oai_chat_response": {
                "choices": [{"finish_reason": "content_filter", "index": 0}]
            },
            "litellm_chat_response": {"finish_reason": "stop"},
        }
        # Should return OpenAI's finish_reason (checked before LiteLLM)
        assert extract_finish_reason(output) == "content_filter"
