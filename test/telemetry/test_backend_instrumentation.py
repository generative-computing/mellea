"""Unit tests for backend_instrumentation helpers — model ID extraction, system name mapping,
context size introspection, and span attribute recording."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from mellea.telemetry.backend_instrumentation import (
    get_context_size,
    get_model_id_str,
    get_system_name,
    record_response_metadata,
    record_token_usage,
)

# --- get_model_id_str ---


@dataclass
class _BackendWithStrModelId:
    model_id: str


@dataclass
class _HFModelId:
    hf_model_name: str


@dataclass
class _BackendWithHFModelId:
    model_id: _HFModelId


def test_get_model_id_str_plain_string():
    backend = _BackendWithStrModelId(model_id="granite-3-8b")
    assert get_model_id_str(backend) == "granite-3-8b"


def test_get_model_id_str_hf_model_name():
    backend = _BackendWithHFModelId(
        model_id=_HFModelId(hf_model_name="ibm-granite/granite-4.0-micro")
    )
    assert get_model_id_str(backend) == "ibm-granite/granite-4.0-micro"


def test_get_model_id_str_no_model_id_returns_class_name():
    class UnknownBackend:
        pass

    backend = UnknownBackend()
    assert get_model_id_str(backend) == "UnknownBackend"


# --- get_system_name ---


def _fake_backend(class_name: str) -> object:
    return type(class_name, (), {})()


def test_get_system_name_openai():
    assert get_system_name(_fake_backend("OpenAIBackend")) == "openai"


def test_get_system_name_ollama():
    assert get_system_name(_fake_backend("OllamaModelBackend")) == "ollama"


def test_get_system_name_huggingface():
    assert get_system_name(_fake_backend("LocalHFBackend")) == "huggingface"


def test_get_system_name_hf_shortname():
    assert get_system_name(_fake_backend("HFBackend")) == "huggingface"


def test_get_system_name_watsonx():
    assert get_system_name(_fake_backend("WatsonxBackend")) == "watsonx"


def test_get_system_name_litellm():
    assert get_system_name(_fake_backend("LiteLLMBackend")) == "litellm"


def test_get_system_name_unknown_returns_class_name():
    backend = _fake_backend("SomeCustomBackend")
    assert get_system_name(backend) == "SomeCustomBackend"


# --- get_context_size ---


def test_get_context_size_with_len():
    ctx = [1, 2, 3]
    assert get_context_size(ctx) == 3


def test_get_context_size_empty_list():
    assert get_context_size([]) == 0


def test_get_context_size_with_turns():
    ctx = type("Ctx", (), {"turns": [1, 2, 3, 4]})()
    assert get_context_size(ctx) == 4


def test_get_context_size_no_len_no_turns():
    class Opaque:
        pass

    assert get_context_size(Opaque()) == 0


def test_get_context_size_len_raises_returns_zero():
    class Broken:
        def __len__(self):
            raise RuntimeError("broken")

    assert get_context_size(Broken()) == 0


# --- record_token_usage ---


def _mock_span():
    return MagicMock()


def test_record_token_usage_from_dict():
    span = _mock_span()
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    record_token_usage(span, usage)
    calls = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert calls.get("gen_ai.usage.input_tokens") == 10
    assert calls.get("gen_ai.usage.output_tokens") == 20
    assert calls.get("gen_ai.usage.total_tokens") == 30


def test_record_token_usage_from_object():
    span = _mock_span()
    usage = type(
        "Usage", (), {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}
    )()
    record_token_usage(span, usage)
    calls = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert calls.get("gen_ai.usage.input_tokens") == 5


def test_record_token_usage_none_span_no_op():
    # Should not raise
    record_token_usage(None, {"prompt_tokens": 1})


def test_record_token_usage_none_usage_no_op():
    span = _mock_span()
    record_token_usage(span, None)
    span.set_attribute.assert_not_called()


def test_record_token_usage_partial_fields():
    span = _mock_span()
    usage = {"prompt_tokens": 7}
    record_token_usage(span, usage)
    calls = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert calls.get("gen_ai.usage.input_tokens") == 7
    assert "gen_ai.usage.output_tokens" not in calls


# --- record_response_metadata ---


def test_record_response_metadata_model_from_dict():
    span = _mock_span()
    response = {"model": "granite-3-8b", "choices": [], "id": "resp-123"}
    record_response_metadata(span, response)
    calls = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert calls.get("gen_ai.response.model") == "granite-3-8b"
    assert calls.get("gen_ai.response.id") == "resp-123"


def test_record_response_metadata_explicit_model_id_overrides():
    span = _mock_span()
    response = {"model": "old-model"}
    record_response_metadata(span, response, model_id="new-model")
    calls = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert calls.get("gen_ai.response.model") == "new-model"


def test_record_response_metadata_finish_reason():
    span = _mock_span()
    response = {"choices": [{"finish_reason": "stop"}]}
    record_response_metadata(span, response)
    calls = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert calls.get("gen_ai.response.finish_reasons") == ["stop"]


def test_record_response_metadata_none_span_no_op():
    record_response_metadata(None, {"model": "x"})


def test_record_response_metadata_none_response_no_op():
    span = _mock_span()
    record_response_metadata(span, None)
    span.set_attribute.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
