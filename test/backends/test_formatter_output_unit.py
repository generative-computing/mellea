# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for how a `FormatterBackend` applies its formatter to model output.

Once a generation completes, the backend's formatter extracts the response from
the generated text (`Formatter.parse_output`): the response becomes `mot.value`,
the generated text is kept on `mot.raw.text`, and the originating action parses the
response. Exercised with `LLMONFormatter`, whose replies can carry LLMON markup, on
a stub backend standing in for `LocalHFBackend`, the one backend that supports LLMON;
every other backend raises for an LLMON model or formatter.
"""

from typing import Any

import pydantic
import pytest

pytest.importorskip(
    "llmon.lang.json.llmon_converters", reason='install "mellea[llmon]"'
)

from mellea.backends.backend import FormatterBackend
from mellea.backends.model_ids import ModelIdentifier
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

    # Stands in for `LocalHFBackend`, the one backend that supports LLMON.
    _supports_llmon = True

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
    backend_cls = getattr(pytest.importorskip(module), cls)
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
            "llmon95:latest",
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
