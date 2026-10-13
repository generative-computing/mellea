# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for how a `FormatterBackend` applies its formatter to model output.

Once a generation completes, the backend's formatter extracts the response from
the generated text (`Formatter.parse_output`): the response becomes `mot.value`,
the generated text is kept on `mot.raw.text`, and the originating action parses the
response. Exercised with `LLMONFormatter`, whose replies can carry LLMON markup, on
a stub backend and on `OpenAIBackend` with a mocked client.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pydantic
import pytest

pytest.importorskip("lang.json.llmon_converters", reason='install "mellea[llmon]"')

from openai.types import Completion
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_choice import CompletionChoice

from mellea.backends import ModelOption
from mellea.backends.backend import FormatterBackend
from mellea.backends.openai import OpenAIBackend
from mellea.backends.tools import MelleaTool
from mellea.backends.utils import to_tool_calls
from mellea.core import GenerateLog, GenerateType, ModelOutputThunk, ModelToolCall
from mellea.formatters import ChatFormatter, LLMONFormatter, TemplateFormatter
from mellea.stdlib.components import Instruction, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.functional import acall_tools
from mellea.stdlib.sampling import RejectionSamplingStrategy

_REPLY = (
    "<|open|>assistant<|:|>2921<|close|>the answer"
    "<|open_end|>assistant<|:|>2921<|close|>"
)


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression.

    Args:
        expression: The expression to evaluate.
    """
    return ""


async def _append(mot: ModelOutputThunk, chunk: Any) -> None:
    mot._underlying_value = (mot._underlying_value or "") + chunk


async def _log_generation(mot: ModelOutputThunk) -> None:
    # Real backends attach a generate log in post-processing; sampling requires one.
    log = GenerateLog()
    log.action = mot._call.action
    mot._generate_log = log


class _StubBackend(FormatterBackend):
    """Streams a fixed reply, in chunks, for whatever it is asked."""

    def __init__(self, formatter: ChatFormatter, chunks: list[str]):
        super().__init__("stub-model", formatter)
        self._model_id = "stub-model"
        self._provider = "stub"
        self._chunks = chunks

    async def _generate_from_context(
        self, action, ctx, *, format=None, model_options=None, tool_calls=False
    ):
        mot = ModelOutputThunk(value=None)
        mot._call.action = action
        mot._gen.generate_type = GenerateType.ASYNC
        mot._gen.process = _append
        mot._gen.post_process = _log_generation
        mot._gen.chunk_size = 0
        for chunk in self._chunks:
            mot._gen.queue.put_nowait(chunk)
        mot._gen.queue.put_nowait(None)
        return mot, ctx.add(action).add(mot)

    async def _generate_from_raw(self, actions, ctx, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _llmon_stub(chunks: list[str] | None = None) -> _StubBackend:
    return _StubBackend(LLMONFormatter("llmon95:latest"), chunks or [_REPLY])


class TestStubBackend:
    async def test_value_is_the_extracted_reply_and_the_generated_text_is_kept(self):
        mot, _ = await _llmon_stub().generate_from_context(
            Instruction("Answer."), ChatContext()
        )
        assert await mot.avalue() == "the answer"
        assert mot.raw.text == _REPLY
        assert mot.parsed_repr == "the answer"

    async def test_streamed_deltas_are_the_generated_text(self):
        chunks = [_REPLY[:20], _REPLY[20:45], _REPLY[45:]]
        mot, _ = await _llmon_stub(chunks).generate_from_context(
            Instruction("Answer."), ChatContext()
        )
        deltas = []
        while not mot.is_computed():
            deltas.append(await mot.astream())
        assert "".join(deltas) == _REPLY
        assert mot.value == "the answer"

    async def test_output_is_untouched_by_a_template_formatter(self):
        backend = _StubBackend(TemplateFormatter(model_id="stub-model"), [_REPLY])
        mot, _ = await backend.generate_from_context(
            Instruction("Answer."), ChatContext()
        )
        assert await mot.avalue() == _REPLY
        assert mot.raw.text is None

    async def test_sampling_reparse_keeps_the_extracted_reply(self):
        # Rejection sampling re-parses each result with its action; that parse must
        # see the extracted reply, not the generated markup.
        result = await RejectionSamplingStrategy(loop_budget=1).sample(
            Instruction("Answer."),
            context=ChatContext(),
            backend=_llmon_stub(),
            requirements=[],
        )
        assert result.result.value == "the answer"
        assert result.result.parsed_repr == "the answer"

    async def test_called_tool_output_is_sent_as_plain_text(self):
        # `call_tools` prints each tool's output with the backend's formatter; the
        # chat template wraps the tool message in its own tool response block.
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
            ModelOutputThunk("", tool_calls=[call]), _llmon_stub()
        )
        assert message.role == "tool"
        assert message.content == "248171"

    async def test_format_is_accepted_for_an_llmon_model(self):
        class _Answer(pydantic.BaseModel):
            text: str

        mot, _ = await _llmon_stub().generate_from_context(
            Instruction("Answer."), ChatContext(), format=_Answer
        )
        assert await mot.avalue() == "the answer"


def _openai_backend(**kwargs: Any) -> OpenAIBackend:
    return OpenAIBackend(
        model_id="llmon95:latest",
        base_url="http://localhost:9999/v1",
        api_key="test-key",
        **kwargs,
    )


def _completion(message: ChatCompletionMessage) -> ChatCompletion:
    return ChatCompletion(
        id="t",
        choices=[Choice(finish_reason="stop", index=0, message=message)],
        created=0,
        model="llmon95:latest",
        object="chat.completion",
    )


class TestOpenAIBackend:
    def test_llmon_model_defaults_to_the_llmon_formatter(self):
        assert isinstance(_openai_backend().formatter, LLMONFormatter)
        explicit = TemplateFormatter(model_id="llmon95:latest")
        assert _openai_backend(formatter=explicit).formatter is explicit

    async def test_chat_round_trip(self):
        backend = _openai_backend()
        with patch.object(
            backend._async_client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _completion(
                ChatCompletionMessage(role="assistant", content=_REPLY)
            )
            mot, _ = await backend.generate_from_context(
                Instruction("Summarize."),
                ChatContext(),
                model_options={ModelOption.STREAM: False},
            )
            await mot.avalue()

        (sent,) = mock_create.call_args.kwargs["messages"]
        assert sent["content"] == "Summarize."
        assert mot.value == "the answer"
        assert mot.raw.text == _REPLY

    async def test_instruction_fields_raise_before_the_request_is_sent(self):
        # OpenAI-compatible servers such as vLLM drop unknown message fields, so a
        # turn that needs the chat template's fields fails rather than losing them.
        backend = _openai_backend()
        instruction = Instruction("Summarize.", grounding_context={"doc": "Text."})
        with patch.object(
            backend._async_client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            with pytest.raises(ValueError, match="huggingface"):
                await backend.generate_from_context(
                    instruction,
                    ChatContext(),
                    model_options={ModelOption.STREAM: False},
                )
        mock_create.assert_not_called()

    async def test_chat_reply_message_holds_only_the_reply_text(self):
        # A chat turn parses into a Message built from the provider's response; its
        # content must be the extracted reply, which is also what is sent back.
        backend = _openai_backend()
        with patch.object(
            backend._async_client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _completion(
                ChatCompletionMessage(role="assistant", content=_REPLY)
            )
            mot, _ = await backend.generate_from_context(
                Message("user", "Hi."),
                ChatContext(),
                model_options={ModelOption.STREAM: False},
            )
            await mot.avalue()

        assert isinstance(mot.parsed_repr, Message)
        assert mot.parsed_repr.content == "the answer"
        (resent,) = backend.formatter.to_chat_messages([mot])
        assert resent.content == "the answer"

    async def test_tool_call_round_trip(self):
        backend = _openai_backend()
        call = ChatCompletionMessageFunctionToolCall(
            id="c1",
            type="function",
            function={"name": "calculator", "arguments": '{"expression": "847 * 293"}'},
        )
        with patch.object(
            backend._async_client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _completion(
                ChatCompletionMessage(role="assistant", content=None, tool_calls=[call])
            )
            mot, _ = await backend.generate_from_context(
                Message("user", "What is 847 * 293?"),
                ChatContext(),
                model_options={
                    ModelOption.STREAM: False,
                    ModelOption.TOOLS: [MelleaTool.from_callable(calculator)],
                },
                tool_calls=True,
            )
            await mot.avalue()

        # The tool definitions go to the server, whose chat template renders them.
        (tool,) = mock_create.call_args.kwargs["tools"]
        assert tool["function"]["name"] == "calculator"
        assert mot.tool_calls is not None
        assert [(c.name, c.args) for c in mot.tool_calls] == [
            ("calculator", {"expression": "847 * 293"})
        ]
        # Sent back, the assistant turn carries the call for the chat template to
        # write out, and the tool result is its plain text.
        assistant, result = backend.formatter.to_chat_messages(
            [mot, Message("tool", "248171", tool_call_id="c1")]
        )
        assert assistant.content == ""
        assert assistant.tool_calls is not None
        assert assistant.tool_calls[0]["function"]["name"] == "calculator"
        assert result.content == "248171"

    async def test_raw_round_trip(self):
        backend = _openai_backend()
        mock_client = MagicMock()
        mock_client.completions.create = AsyncMock(
            return_value=Completion(
                id="raw",
                created=0,
                model="llmon95:latest",
                object="text_completion",
                choices=[CompletionChoice(index=0, finish_reason="stop", text=_REPLY)],
            )
        )
        instruction = Instruction("Answer.")
        with patch.object(
            OpenAIBackend,
            "_async_client",
            new_callable=PropertyMock,
            return_value=mock_client,
        ):
            (result,) = await backend.generate_from_raw([instruction], ChatContext())

        assert mock_client.completions.create.call_args.kwargs["prompt"] == [
            backend.formatter.print(instruction)
        ]
        assert result.value == "the answer"
        assert result.parsed_repr == "the answer"
        assert result.raw.text == _REPLY


def test_tool_call_written_in_the_reply_round_trips_as_tool_calls():
    # On HuggingFace the chat template opens the assistant block, so the model's
    # reply is the tool call as it wrote it, then the closer. The backend finds the
    # call in that text, the extracted reply keeps it, and the next turn sends it as
    # `tool_calls` for the chat template to write back out.
    call = (
        '<tool_call>{"name": "calculator", '
        '"arguments": {"expression": "847 * 293"}}</tool_call>'
    )
    reply = call + "<|open_end|>"
    calls = to_tool_calls({"calculator": MelleaTool.from_callable(calculator)}, reply)
    assert calls is not None
    assert [(c.name, c.args) for c in calls] == [
        ("calculator", {"expression": "847 * 293"})
    ]

    formatter = LLMONFormatter("llmon95:latest")
    mot = ModelOutputThunk(formatter.parse_output(reply), tool_calls=calls)
    assert mot.value == call
    mot.parsed_repr = Message("assistant", mot.value)
    (sent,) = formatter.to_chat_messages([mot])
    assert sent.content == ""
    assert sent.tool_calls is not None
    assert sent.tool_calls[0]["function"]["name"] == "calculator"
