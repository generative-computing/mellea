# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from mellea.backends import ModelOption
from mellea.backends._options import resolve_model_options


def test_resolve_model_options_call_wins_over_helper_and_backend():
    resolved = resolve_model_options(
        backend_defaults={ModelOption.TEMPERATURE: 0.0},
        remap={},
        helper_defaults={ModelOption.TEMPERATURE: 0.5},
        call_options={ModelOption.TEMPERATURE: 0.9},
    )
    assert resolved[ModelOption.TEMPERATURE] == 0.9


def test_resolve_model_options_helper_wins_over_backend_when_no_call_override():
    resolved = resolve_model_options(
        backend_defaults={ModelOption.TEMPERATURE: 0.0},
        remap={},
        helper_defaults={ModelOption.TEMPERATURE: 0.5},
        call_options=None,
    )
    assert resolved[ModelOption.TEMPERATURE] == 0.5


def test_resolve_model_options_backend_used_when_nothing_else_set():
    resolved = resolve_model_options(
        backend_defaults={ModelOption.TEMPERATURE: 0.0}, remap={}, call_options=None
    )
    assert resolved[ModelOption.TEMPERATURE] == 0.0


def test_resolve_model_options_applies_remap_to_backend_and_call_options():
    resolved = resolve_model_options(
        backend_defaults={"temp": 0.0},
        remap={"temp": ModelOption.TEMPERATURE},
        call_options={"temp": 0.7},
    )
    assert resolved == {ModelOption.TEMPERATURE: 0.7}


def test_resolve_model_options_none_call_options_keeps_helper_and_backend_merged():
    resolved = resolve_model_options(
        backend_defaults={ModelOption.CONTEXT_WINDOW: 4096},
        remap={},
        helper_defaults={ModelOption.TEMPERATURE: 0.0},
        call_options=None,
    )
    assert resolved == {ModelOption.CONTEXT_WINDOW: 4096, ModelOption.TEMPERATURE: 0.0}


def test_resolve_model_options_unrelated_keys_are_preserved():
    resolved = resolve_model_options(
        backend_defaults={"backend_only": 1},
        remap={},
        helper_defaults={"helper_only": 2},
        call_options={"call_only": 3},
    )
    assert resolved == {"backend_only": 1, "helper_only": 2, "call_only": 3}


def test_resolve_model_options_extra_body_default_survives_unrelated_call_extra_body():
    """A backend-level extra_body default must survive an unrelated per-call
    extra_body (e.g. an intrinsic/adapter routing call)."""
    resolved = resolve_model_options(
        backend_defaults={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        },
        remap={},
        call_options={"extra_body": {"some_unrelated_field": 123}},
    )
    assert resolved["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "some_unrelated_field": 123,
    }


# ---------------------------------------------------------------------------
# Cross-path delegation tests for merge_extra_body
# ---------------------------------------------------------------------------
# Each scenario is exercised through every path so that a future drift in any
# one path causes exactly the relevant test to fail.
#
# Four path-driver helpers:
#   _via_helper           — ModelOption.merge_extra_body directly
#   _via_merge_model_options — ModelOption.merge_model_options wrapper
#   _via_openai           — OpenAIBackend._merge_user_extra_body
#   _via_litellm          — LiteLLMBackend full pipeline (mocked litellm.acompletion)
# ---------------------------------------------------------------------------

import pytest


def _via_helper(base_eb: dict, over_eb: dict) -> dict:
    """Call ModelOption.merge_extra_body directly and return the result."""
    return ModelOption.merge_extra_body(base_eb, over_eb)


def _via_merge_model_options(base_eb: dict, over_eb: dict) -> dict:
    """Wrap both dicts in {'extra_body': ...}, call merge_model_options, return extra_body."""
    merged = ModelOption.merge_model_options(
        {"extra_body": base_eb}, {"extra_body": over_eb}
    )
    return merged["extra_body"]


def _via_openai(default_eb: dict, base_eb: dict, user_eb: dict | None) -> dict:
    """Construct OpenAIBackend with default_extra_body and call _merge_user_extra_body."""
    pytest.importorskip("openai", reason="openai not installed")
    from mellea.backends.openai import OpenAIBackend

    backend = OpenAIBackend(
        model_id="gpt-4o",
        base_url="http://localhost:9999/v1",
        api_key="test-key",
        default_extra_body=default_eb,
    )
    return backend._merge_user_extra_body(base_eb, user_eb)


async def _via_litellm(default_eb: dict, user_eb: dict) -> dict:
    """Run LiteLLMBackend._generate_from_chat_context_standard with mocked acompletion.

    Returns the extra_body value from the wire-call kwargs.
    """
    pytest.importorskip("litellm", reason="litellm not installed")
    from unittest.mock import AsyncMock, patch

    from litellm.types.utils import Choices, Message, ModelResponse

    from mellea.backends.litellm import LiteLLMBackend
    from mellea.core import CBlock
    from mellea.stdlib.components import Message as MelleaMessage
    from mellea.stdlib.context import ChatContext

    backend = LiteLLMBackend(
        model_id="hosted_vllm/test",
        base_url="http://localhost:9997",
        model_options={"extra_body": default_eb} if default_eb else {},
    )

    msg = Message(content="ok", role="assistant")
    choice = Choices(finish_reason="stop", index=0, message=msg)
    mock_response = ModelResponse(
        id="test",
        choices=[choice],
        created=0,
        model="hosted_vllm/test",
        object="chat.completion",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    ctx = ChatContext().add(MelleaMessage("user", "Hello"))
    action = CBlock(value="Test")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomplete:
        mock_acomplete.return_value = mock_response
        mot = await backend._generate_from_chat_context_standard(
            action, ctx, model_options={"extra_body": user_eb} if user_eb else {}
        )
        await mot.avalue()

    assert mock_acomplete.called, "litellm.acompletion was never called"
    return dict(mock_acomplete.call_args.kwargs).get("extra_body", {})


# --- Scenario A: deep merge ---


@pytest.mark.asyncio
async def test_merge_extra_body_deep_merge_all_paths():
    """Scenario A: distinct keys from both sides both survive in chat_template_kwargs."""
    base_eb = {"chat_template_kwargs": {"enable_thinking": True}}
    over_eb = {"chat_template_kwargs": {"adapter_name": "foo"}}

    for result in [
        _via_helper(base_eb, over_eb),
        _via_merge_model_options(base_eb, over_eb),
        _via_openai({}, base_eb, over_eb),
        _via_openai(base_eb, {}, over_eb),
    ]:
        ctk = result.get("chat_template_kwargs", {})
        assert ctk.get("enable_thinking") is True, f"enable_thinking missing: {result}"
        assert ctk.get("adapter_name") == "foo", f"adapter_name missing: {result}"

    litellm_result = await _via_litellm(base_eb, over_eb)
    ctk = litellm_result.get("chat_template_kwargs", {})
    assert ctk.get("enable_thinking") is True, (
        f"LiteLLM enable_thinking missing: {litellm_result}"
    )
    assert ctk.get("adapter_name") == "foo", (
        f"LiteLLM adapter_name missing: {litellm_result}"
    )


# --- Scenario B: malformed raises ---


@pytest.mark.asyncio
async def test_merge_extra_body_malformed_raises_all_paths():
    """Scenario B: a non-dict chat_template_kwargs raises TypeError on every path."""
    bad_eb = {"chat_template_kwargs": "not-a-dict"}
    good_eb = {"chat_template_kwargs": {"enable_thinking": True}}

    with pytest.raises(TypeError):
        _via_helper(bad_eb, good_eb)
    with pytest.raises(TypeError):
        _via_helper(good_eb, bad_eb)
    with pytest.raises(TypeError):
        _via_merge_model_options(bad_eb, good_eb)
    with pytest.raises(TypeError):
        _via_merge_model_options(good_eb, bad_eb)
    with pytest.raises(TypeError):
        _via_openai({}, bad_eb, good_eb)
    with pytest.raises(TypeError):
        _via_openai({}, good_eb, bad_eb)
    with pytest.raises(TypeError):
        await _via_litellm(bad_eb, good_eb)
    with pytest.raises(TypeError):
        await _via_litellm(good_eb, bad_eb)


# --- Scenario C: empty chat_template_kwargs → key absent ---


@pytest.mark.asyncio
async def test_merge_extra_body_empty_dropped_all_paths():
    """Scenario C: empty chat_template_kwargs on both sides → key absent from output."""
    empty_eb = {"chat_template_kwargs": {}}

    for result in [
        _via_helper(empty_eb, empty_eb),
        _via_merge_model_options(empty_eb, empty_eb),
        _via_openai({}, empty_eb, empty_eb),
    ]:
        assert "chat_template_kwargs" not in result, (
            f"chat_template_kwargs should be absent but found in: {result}"
        )

    litellm_result = await _via_litellm(empty_eb, empty_eb)
    assert "chat_template_kwargs" not in litellm_result, (
        f"LiteLLM chat_template_kwargs should be absent but found in: {litellm_result}"
    )


# --- Scenario D: three-layer precedence ---


@pytest.mark.asyncio
async def test_merge_extra_body_three_layer_precedence():
    """Scenario D: three-tier merge — distinct keys survive, 'shared' resolves to caller.

    _via_helper and _via_merge_model_options are two-tier and cannot represent the
    default/base/user split directly; only _via_openai and _via_litellm are exercised.
    """
    default_eb = {
        "chat_template_kwargs": {"enable_thinking": True, "shared": "default"}
    }
    base_eb = {"chat_template_kwargs": {"adapter_name": "mellea", "shared": "mellea"}}
    user_eb = {"chat_template_kwargs": {"caller_key": "caller", "shared": "caller"}}

    expected_ctk = {
        "enable_thinking": True,
        "adapter_name": "mellea",
        "caller_key": "caller",
        "shared": "caller",
    }

    openai_result = _via_openai(default_eb, base_eb, user_eb)
    assert openai_result.get("chat_template_kwargs") == expected_ctk, (
        f"OpenAI path: {openai_result}"
    )

    litellm_result = await _via_litellm(default_eb, user_eb)
    # LiteLLM path has no separate "base" tier; it exercises default + user.
    expected_litellm_ctk = {
        "enable_thinking": True,
        "shared": "caller",
        "caller_key": "caller",
    }
    assert litellm_result.get("chat_template_kwargs") == expected_litellm_ctk, (
        f"LiteLLM path: {litellm_result}"
    )
