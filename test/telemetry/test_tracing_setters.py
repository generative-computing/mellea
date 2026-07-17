# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for span-attribute setters in `mellea.telemetry._tracing_setters`."""

from unittest.mock import MagicMock

from mellea.core.base import GenerationMetadata
from mellea.telemetry._tracing_setters import (
    set_attribute_safe,
    set_conversation_id,
    set_mellea_attrs,
    set_request_attrs,
    set_response_attrs,
    set_usage_attrs,
)


def _attrs(span: MagicMock) -> dict:
    return {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}


# set_attribute_safe


def test_set_attribute_safe_none_value_no_op():
    span = MagicMock()
    set_attribute_safe(span, "key", None)
    span.set_attribute.assert_not_called()


def test_set_attribute_safe_bool():
    span = MagicMock()
    set_attribute_safe(span, "flag", True)
    span.set_attribute.assert_called_once_with("flag", True)


def test_set_attribute_safe_int():
    span = MagicMock()
    set_attribute_safe(span, "count", 42)
    span.set_attribute.assert_called_once_with("count", 42)


def test_set_attribute_safe_str():
    span = MagicMock()
    set_attribute_safe(span, "name", "hello")
    span.set_attribute.assert_called_once_with("name", "hello")


def test_set_attribute_safe_list_converted_to_string_list():
    span = MagicMock()
    set_attribute_safe(span, "items", [1, 2, 3])
    span.set_attribute.assert_called_once_with("items", ["1", "2", "3"])


def test_set_attribute_safe_unsupported_type_stringified():
    span = MagicMock()
    set_attribute_safe(span, "obj", {"nested": "dict"})
    span.set_attribute.assert_called_once()
    call_args = span.set_attribute.call_args
    assert call_args.args[0] == "obj"
    assert isinstance(call_args.args[1], str)


# set_request_attrs


def test_set_request_attrs_full():
    span = MagicMock()
    gen = GenerationMetadata(model="gpt-4o", provider="openai")
    set_request_attrs(span, gen, "chat")
    attrs = _attrs(span)
    assert attrs == {
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.operation.name": "chat",
    }


def test_set_request_attrs_operation_always_emitted():
    """Operation is unconditional even when model/provider are unset."""
    span = MagicMock()
    gen = GenerationMetadata(model=None, provider=None)
    set_request_attrs(span, gen, "text_completion")
    attrs = _attrs(span)
    assert attrs == {"gen_ai.operation.name": "text_completion"}


# set_usage_attrs


def test_set_usage_attrs_no_op_when_usage_none():
    span = MagicMock()
    set_usage_attrs(span, None)
    span.set_attribute.assert_not_called()


def test_set_usage_attrs_top_level_token_counts():
    span = MagicMock()
    set_usage_attrs(
        span, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    attrs = _attrs(span)
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.usage.total_tokens"] == 15


def test_set_usage_attrs_cache_read_top_level_preferred():
    """Top-level `cache_read_input_tokens` wins over nested details."""
    span = MagicMock()
    set_usage_attrs(
        span,
        {"cache_read_input_tokens": 99, "prompt_tokens_details": {"cached_tokens": 1}},
    )
    attrs = _attrs(span)
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 99


def test_set_usage_attrs_cache_read_falls_back_to_details():
    span = MagicMock()
    set_usage_attrs(span, {"prompt_tokens_details": {"cached_tokens": 75}})
    attrs = _attrs(span)
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 75


def test_set_usage_attrs_cache_creation():
    span = MagicMock()
    set_usage_attrs(span, {"cache_creation_input_tokens": 200})
    attrs = _attrs(span)
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 200


def test_set_usage_attrs_reasoning_top_level_preferred():
    span = MagicMock()
    set_usage_attrs(
        span,
        {"reasoning_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 1}},
    )
    attrs = _attrs(span)
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 50


def test_set_usage_attrs_reasoning_falls_back_to_details():
    span = MagicMock()
    set_usage_attrs(span, {"completion_tokens_details": {"reasoning_tokens": 30}})
    attrs = _attrs(span)
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 30


# set_response_attrs


def test_set_response_attrs_full():
    span = MagicMock()
    gen = GenerationMetadata(
        model="m",
        provider="p",
        response_model="gpt-4o-2024-08-06",
        response_id="chatcmpl-abc",
        finish_reasons=["stop"],
    )
    set_response_attrs(span, gen)
    attrs = _attrs(span)
    assert attrs["gen_ai.response.model"] == "gpt-4o-2024-08-06"
    assert attrs["gen_ai.response.id"] == "chatcmpl-abc"
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]


def test_set_response_attrs_skips_unset_fields():
    span = MagicMock()
    gen = GenerationMetadata(model="m", provider="p")
    set_response_attrs(span, gen)
    span.set_attribute.assert_not_called()


def test_set_response_attrs_finish_reasons_converted_to_list():
    """Tuples and other iterables are normalized to list before emission."""
    span = MagicMock()
    gen = GenerationMetadata(model="m", provider="p", finish_reasons=("stop", "length"))
    set_response_attrs(span, gen)
    attrs = _attrs(span)
    assert attrs["gen_ai.response.finish_reasons"] == ["stop", "length"]
    assert isinstance(attrs["gen_ai.response.finish_reasons"], list)


# set_mellea_attrs


def test_set_mellea_attrs_action_class_name_from_action():
    span = MagicMock()

    class MyAction:
        pass

    mot = MagicMock()
    mot._call.action = MyAction()
    mot._call.context = None
    gen = GenerationMetadata(model="m", provider="p")
    set_mellea_attrs(span, mot, gen)
    attrs = _attrs(span)
    assert attrs["mellea.action_type"] == "MyAction"


def test_set_mellea_attrs_skips_action_type_when_no_action():
    span = MagicMock()
    mot = MagicMock(spec=[])  # No _action attribute
    gen = GenerationMetadata(model="m", provider="p")
    set_mellea_attrs(span, mot, gen)
    attrs = _attrs(span)
    assert "mellea.action_type" not in attrs


def test_set_mellea_attrs_context_size():
    """Length when context is non-empty; zero when falsy."""
    gen = GenerationMetadata(model="m", provider="p")

    span = MagicMock()
    mot = MagicMock()
    mot._call.action = None
    mot._call.context = [1, 2, 3]
    set_mellea_attrs(span, mot, gen)
    assert _attrs(span)["mellea.context_size"] == 3

    span = MagicMock()
    mot._call.context = None
    set_mellea_attrs(span, mot, gen)
    assert _attrs(span)["mellea.context_size"] == 0


def test_set_mellea_attrs_streaming_conditional():
    """`mellea.streaming` is emitted only when streaming is truthy."""
    mot = MagicMock()
    mot._call.action = None
    mot._call.context = None

    span = MagicMock()
    set_mellea_attrs(
        span, mot, GenerationMetadata(model="m", provider="p", streaming=True)
    )
    assert _attrs(span)["mellea.streaming"] is True

    span = MagicMock()
    set_mellea_attrs(
        span, mot, GenerationMetadata(model="m", provider="p", streaming=False)
    )
    assert "mellea.streaming" not in _attrs(span)


# set_conversation_id


def test_set_conversation_id_emits_when_session_id_set():
    from mellea.telemetry.context import with_context

    span = MagicMock()
    with with_context(session_id="sess-123"):
        set_conversation_id(span)
    attrs = _attrs(span)
    assert attrs["gen_ai.conversation.id"] == "sess-123"


def test_set_conversation_id_no_op_when_session_id_unset():
    span = MagicMock()
    set_conversation_id(span)
    span.set_attribute.assert_not_called()
