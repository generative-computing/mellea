# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLMON on backends.

Only `LocalHFBackend` supports LLMON, so every other backend raises for an LLMON
model or an `LLMONFormatter`. `LocalHFBackend` decodes a reply without special
tokens, so the reply is the model's text, and a tool call the model wrote in it goes
back to the chat template as the turn's `tool_calls`. A called tool's output is sent
as plain text.
"""

from typing import Any

import pytest

pytest.importorskip(
    "llmon.lang.json.llmon_converters", reason='install "mellea[llmon]"'
)

from mellea.backends.backend import FormatterBackend
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.tools import MelleaTool
from mellea.backends.utils import to_tool_calls
from mellea.core import ModelOutputThunk, ModelToolCall
from mellea.formatters import ChatFormatter, LLMONFormatter, TemplateFormatter
from mellea.stdlib.components import Message
from mellea.stdlib.functional import acall_tools


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression.

    Args:
        expression: The expression to evaluate.
    """
    return ""


class _StubBackend(FormatterBackend):
    """A backend that supports LLMON, as `LocalHFBackend` does, and generates nothing."""

    _supports_llmon = True

    def __init__(self, formatter: ChatFormatter):
        super().__init__("stub-model", formatter)

    async def _generate_from_context(self, action, ctx, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def _generate_from_raw(self, actions, ctx, **kwargs):  # pragma: no cover
        raise NotImplementedError


_OTHER_BACKENDS = {
    "openai": ("mellea.backends.openai", "OpenAIBackend"),
    "ollama": ("mellea.backends.ollama", "OllamaModelBackend"),
    "litellm": ("mellea.backends.litellm", "LiteLLMBackend"),
    "watsonx": ("mellea.backends.watsonx", "WatsonxAIBackend"),
}


def _other_backend(
    name: str, model_id: str | ModelIdentifier, formatter: ChatFormatter | None = None
) -> FormatterBackend:
    module, cls = _OTHER_BACKENDS[name]
    # Mellea raises its own ImportError for a backend whose extra is missing.
    backend_cls = getattr(pytest.importorskip(module, exc_type=ImportError), cls)
    kwargs: dict[str, Any] = {}
    if name == "openai":
        kwargs = {"base_url": "http://localhost:9999/v1", "api_key": "test-key"}
    return backend_cls(model_id=model_id, formatter=formatter, **kwargs)


@pytest.mark.parametrize("name", list(_OTHER_BACKENDS))
class TestOtherBackends:
    """Only `LocalHFBackend` supports LLMON, so every other backend raises for an LLMON
    model or formatter before it reaches its server."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "ibm-granite/granite-4.1-3b-llmon-preview",
            # The preview model as a vLLM server serves it, under its HF name.
            ModelIdentifier(
                hf_model_name="ibm-granite/granite-4.1-3b-llmon-preview",
                openai_name="ibm-granite/granite-4.1-3b-llmon-preview",
            ),
        ],
    )
    def test_llmon_model_raises(self, name, model_id):
        with pytest.raises(ValueError, match="only LocalHFBackend supports"):
            _other_backend(name, model_id)

    def test_llmon_formatter_raises(self, name):
        formatter = LLMONFormatter("granite4:micro")
        with pytest.raises(ValueError, match="only be used with LocalHFBackend"):
            _other_backend(name, "granite4:micro", formatter)


def test_other_models_default_to_a_template_formatter():
    backend = _other_backend("openai", "granite4:micro")
    assert isinstance(backend.formatter, TemplateFormatter)
    assert not isinstance(backend.formatter, LLMONFormatter)


async def test_called_tool_output_is_sent_as_plain_text():
    # `call_tools` prints each tool's output with the backend's formatter; the chat
    # template wraps the tool message in its own tool response block.
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers.

        Args:
            a: The first number.
            b: The second number.
        """
        return a * b

    call = ModelToolCall(
        "multiply", MelleaTool.from_callable(multiply), {"a": 847, "b": 293}
    )
    (message,) = await acall_tools(
        ModelOutputThunk("", tool_calls=[call]),
        _StubBackend(LLMONFormatter("ibm-granite/granite-4.1-3b-llmon-preview")),
    )
    assert message.role == "tool"
    assert message.content == "248171"


def test_tool_call_written_in_the_reply_round_trips_as_tool_calls():
    # The chat template opens the assistant block, and the HuggingFace backend decodes
    # the reply without special tokens, so the reply is the tool call as the model
    # wrote it. The backend finds the call in that text, and the next turn sends it as
    # `tool_calls` for the chat template to write back out.
    call = (
        '<tool_call>{"name": "calculator", '
        '"arguments": {"expression": "847 * 293"}}</tool_call>'
    )
    calls = to_tool_calls({"calculator": MelleaTool.from_callable(calculator)}, call)
    assert calls is not None
    assert [(c.name, c.args) for c in calls] == [
        ("calculator", {"expression": "847 * 293"})
    ]

    mot = ModelOutputThunk(call, tool_calls=calls)
    mot.parsed_repr = Message("assistant", call)
    (sent,) = LLMONFormatter(
        "ibm-granite/granite-4.1-3b-llmon-preview"
    ).to_chat_messages([mot])
    assert sent.content == ""
    assert sent.tool_calls is not None
    assert sent.tool_calls[0]["function"]["name"] == "calculator"
