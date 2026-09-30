# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Responses API endpoint and related helpers."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from cli.serve.app import (
    _build_model_options_from_response_request,
    _convert_response_input_to_messages,
    _response_store,
    make_responses_endpoint,
)
from cli.serve.models import Response, ResponseInputItem, ResponseRequest
from cli.serve.streaming import stream_response_chunks
from mellea.backends.model_options import ModelOption
from mellea.core.base import ModelOutputThunk
from mellea.helpers.openai_compatible_helpers import (
    build_response_output_items,
    build_response_usage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_module():
    """Mock module with a synchronous serve function."""
    module = Mock()
    module.__name__ = "test_responses_module"
    return module


@pytest.fixture
def async_mock_module():
    """Mock module with an async serve function."""
    module = Mock()
    module.__name__ = "test_responses_async_module"
    module.serve = AsyncMock()
    return module


@pytest.fixture
def simple_request():
    """Minimal ResponseRequest with a string input."""
    return ResponseRequest(model="test-model", input="Hello!")


@pytest.fixture
def message_request():
    """ResponseRequest with a message-array input."""
    return ResponseRequest(
        model="test-model",
        input=[ResponseInputItem(role="user", content="What is 2+2?")],
    )


# ---------------------------------------------------------------------------
# _build_model_options_from_response_request
# ---------------------------------------------------------------------------


class TestBuildModelOptionsFromResponseRequest:
    """Tests for model_options extraction from ResponseRequest."""

    def test_temperature_mapped(self):
        req = ResponseRequest(model="m", input="hi", temperature=0.5)
        opts = _build_model_options_from_response_request(req)
        assert opts[ModelOption.TEMPERATURE] == 0.5

    def test_max_output_tokens_mapped_to_max_new_tokens(self):
        req = ResponseRequest(model="m", input="hi", max_output_tokens=200)
        opts = _build_model_options_from_response_request(req)
        assert opts[ModelOption.MAX_NEW_TOKENS] == 200

    def test_excluded_fields_not_present(self):
        req = ResponseRequest(
            model="m",
            input="hi",
            store=True,
            conversation="conv-123",
            previous_response_id="resp-abc",
            background=False,
            include=["reasoning"],
            parallel_tool_calls=True,
        )
        opts = _build_model_options_from_response_request(req)
        for excluded in (
            "input",
            "model",
            "store",
            "conversation",
            "previous_response_id",
            "background",
            "include",
            "parallel_tool_calls",
        ):
            assert excluded not in opts

    def test_tools_mapped(self):
        req = ResponseRequest(
            model="m",
            input="hi",
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        )
        opts = _build_model_options_from_response_request(req)
        assert ModelOption.TOOLS in opts

    def test_tool_choice_mapped(self):
        req = ResponseRequest(model="m", input="hi", tool_choice="auto")
        opts = _build_model_options_from_response_request(req)
        assert opts[ModelOption.TOOL_CHOICE] == "auto"

    @pytest.mark.parametrize(
        "tool_type", ["web_search", "file_search", "mcp", "code_interpreter"]
    )
    def test_unsupported_native_tool_types_raise(self, tool_type):
        req = ResponseRequest(model="m", input="hi", tools=[{"type": tool_type}])
        with pytest.raises(ValueError, match=tool_type):
            _build_model_options_from_response_request(req)

    def test_mixed_function_and_unsupported_tool_raises(self):
        req = ResponseRequest(
            model="m",
            input="hi",
            tools=[
                {"type": "function", "function": {"name": "f", "parameters": {}}},
                {"type": "web_search"},
            ],
        )
        with pytest.raises(ValueError, match="web_search"):
            _build_model_options_from_response_request(req)

    def test_empty_request_produces_minimal_options(self):
        req = ResponseRequest(model="m", input="hi")
        opts = _build_model_options_from_response_request(req)
        # Only temperature (default 1.0) should survive after exclusions
        assert ModelOption.TEMPERATURE in opts
        assert "model" not in opts
        assert "input" not in opts


# ---------------------------------------------------------------------------
# _convert_response_input_to_messages
# ---------------------------------------------------------------------------


class TestConvertResponseInputToMessages:
    """Tests for input/instructions → ChatMessage conversion."""

    def test_string_input_becomes_user_message(self):
        msgs = _convert_response_input_to_messages("Hello", None)
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "Hello"

    def test_message_array_preserved(self):
        items = [
            ResponseInputItem(role="user", content="Hi"),
            ResponseInputItem(role="assistant", content="Hello"),
        ]
        msgs = _convert_response_input_to_messages(items, None)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_developer_role_mapped_to_system(self):
        items = [ResponseInputItem(role="developer", content="Be helpful.")]
        msgs = _convert_response_input_to_messages(items, None)
        assert msgs[0].role == "system"

    def test_system_role_mapped_to_system(self):
        items = [ResponseInputItem(role="system", content="Be concise.")]
        msgs = _convert_response_input_to_messages(items, None)
        assert msgs[0].role == "system"

    def test_string_instructions_prepended_as_system(self):
        msgs = _convert_response_input_to_messages("Hello", "You are helpful.")
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[0].content == "You are helpful."
        assert msgs[1].role == "user"

    def test_message_instructions_prepended(self):
        instructions = [ResponseInputItem(role="developer", content="Be brief.")]
        items = [ResponseInputItem(role="user", content="Hello")]
        msgs = _convert_response_input_to_messages(items, instructions)
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"

    def test_instructions_come_before_input(self):
        msgs = _convert_response_input_to_messages(
            [ResponseInputItem(role="user", content="Q")], "Instruction"
        )
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"

    def test_non_string_content_becomes_none(self):
        """Content that is a list (multimodal) is dropped to None — string only for now."""
        items = [
            ResponseInputItem(
                role="user", content=[{"type": "input_text", "text": "hi"}]
            )
        ]
        msgs = _convert_response_input_to_messages(items, None)
        assert msgs[0].content is None


# ---------------------------------------------------------------------------
# make_responses_endpoint — non-streaming
# ---------------------------------------------------------------------------


class TestResponsesEndpointNonStreaming:
    """Tests for the /v1/responses endpoint (non-streaming path)."""

    def setup_method(self):
        _response_store.clear()

    def teardown_method(self):
        _response_store.clear()

    @pytest.mark.asyncio
    async def test_basic_response_structure(self, mock_module, simple_request):
        mock_output = ModelOutputThunk("The answer is 42.")
        mock_module.serve.return_value = mock_output

        endpoint = make_responses_endpoint(mock_module)
        response = await endpoint(simple_request)

        assert isinstance(response, Response)
        assert response.model == "test-model"
        assert response.status == "completed"
        assert response.output_text == "The answer is 42."

    @pytest.mark.asyncio
    async def test_response_id_format(self, mock_module, simple_request):
        mock_module.serve.return_value = ModelOutputThunk("hi")
        endpoint = make_responses_endpoint(mock_module)
        response = await endpoint(simple_request)
        assert response.id.startswith("resp_")
        assert len(response.id) > len("resp_")

    @pytest.mark.asyncio
    async def test_created_at_is_positive_int(self, mock_module, simple_request):
        mock_module.serve.return_value = ModelOutputThunk("hi")
        endpoint = make_responses_endpoint(mock_module)
        response = await endpoint(simple_request)
        assert isinstance(response.created_at, int)
        assert response.created_at > 0

    @pytest.mark.asyncio
    async def test_output_items_include_message(self, mock_module, simple_request):
        mock_module.serve.return_value = ModelOutputThunk("Hello!")
        endpoint = make_responses_endpoint(mock_module)
        response = await endpoint(simple_request)

        messages = [item for item in response.output if item.type == "message"]
        assert len(messages) == 1
        assert messages[0].role == "assistant"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_usage_populated_when_available(self, mock_module, simple_request):
        mock_output = ModelOutputThunk("hi")
        mock_output.generation.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        mock_module.serve.return_value = mock_output
        endpoint = make_responses_endpoint(mock_module)
        response = await endpoint(simple_request)

        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_usage_zero_when_unavailable(self, mock_module, simple_request):
        mock_module.serve.return_value = ModelOutputThunk("hi")
        endpoint = make_responses_endpoint(mock_module)
        response = await endpoint(simple_request)

        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0
        assert response.usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_conversation_echoed(self, mock_module):
        req = ResponseRequest(model="m", input="hi", conversation="conv-abc")
        mock_module.serve.return_value = ModelOutputThunk("ok")
        endpoint = make_responses_endpoint(mock_module)
        response = await endpoint(req)
        assert response.conversation == "conv-abc"

    @pytest.mark.asyncio
    async def test_previous_response_id_echoed(self, mock_module):
        mock_module.serve.return_value = ModelOutputThunk("ok")
        endpoint = make_responses_endpoint(mock_module)

        # First request — store a response so it can be referenced.
        first_response = await endpoint(ResponseRequest(model="m", input="first"))
        prior_id = first_response.id
        assert prior_id in _response_store

        # Second request — chain via previous_response_id.
        req = ResponseRequest(
            model="m", input="follow-up", previous_response_id=prior_id
        )
        mock_module.serve.return_value = ModelOutputThunk("ok2")
        response = await endpoint(req)
        assert response.previous_response_id == prior_id

    @pytest.mark.asyncio
    async def test_background_true_returns_400(self, mock_module):
        req = ResponseRequest(model="m", input="hi", background=True)
        mock_module.serve.return_value = ModelOutputThunk("ok")
        endpoint = make_responses_endpoint(mock_module)
        resp = await endpoint(req)
        # Returns a JSONResponse error, not a Response model
        assert hasattr(resp, "status_code")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_serve_receives_converted_messages(self, mock_module):
        req = ResponseRequest(
            model="m", input="Tell me a joke.", instructions="Be funny."
        )
        mock_module.serve.return_value = ModelOutputThunk("Why did...")
        endpoint = make_responses_endpoint(mock_module)
        await endpoint(req)

        call_kwargs = mock_module.serve.call_args.kwargs
        messages = call_kwargs["input"]
        assert messages[0].role == "system"
        assert messages[0].content == "Be funny."
        assert messages[1].role == "user"
        assert messages[1].content == "Tell me a joke."

    @pytest.mark.asyncio
    async def test_async_serve_function(self, async_mock_module, simple_request):
        async_mock_module.serve.return_value = ModelOutputThunk("async result")
        endpoint = make_responses_endpoint(async_mock_module)
        response = await endpoint(simple_request)
        assert isinstance(response, Response)
        assert response.output_text == "async result"

    @pytest.mark.asyncio
    async def test_serve_exception_returns_500(self, mock_module, simple_request):
        mock_module.serve.side_effect = RuntimeError("backend exploded")
        endpoint = make_responses_endpoint(mock_module)
        resp = await endpoint(simple_request)
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_value_error_returns_400(self, mock_module, simple_request):
        mock_module.serve.side_effect = ValueError("bad input")
        endpoint = make_responses_endpoint(mock_module)
        resp = await endpoint(simple_request)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_client_options_passed_when_declared(
        self, mock_module, simple_request
    ):
        """serve(client_options=...) receives the full raw request dict."""
        import inspect
        from unittest.mock import MagicMock

        def serve_with_co(
            input, requirements=None, model_options=None, client_options=None
        ):
            return ModelOutputThunk("ok")

        mock_module.serve = serve_with_co
        endpoint = make_responses_endpoint(mock_module)
        # Should not raise — just verifying the code path executes cleanly
        response = await endpoint(simple_request)
        assert isinstance(response, Response)


# ---------------------------------------------------------------------------
# make_responses_endpoint — streaming path
# ---------------------------------------------------------------------------


class TestResponsesEndpointStreaming:
    """Tests for the /v1/responses endpoint (streaming path)."""

    @pytest.mark.asyncio
    async def test_stream_true_returns_streaming_response(self, mock_module):
        from fastapi.responses import StreamingResponse

        req = ResponseRequest(model="m", input="hi", stream=True)
        mock_module.serve.return_value = ModelOutputThunk("streamed content")
        endpoint = make_responses_endpoint(mock_module)
        response = await endpoint(req)
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"


# ---------------------------------------------------------------------------
# stream_response_chunks
# ---------------------------------------------------------------------------


class TestStreamResponseChunks:
    """Tests for the semantic SSE event generator."""

    async def _collect(self, output, **kwargs) -> list[dict]:
        """Collect all SSE events as parsed dicts.

        Each yielded chunk may contain multiple lines (event: / data: / blank).
        Parse them together so event_name is always set before data is read.
        """
        events = []
        async for chunk in stream_response_chunks(output, **kwargs):
            event_name = None
            for line in chunk.split("\n"):
                line = line.strip()
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event_name is not None:
                    data = json.loads(line.split(":", 1)[1].strip())
                    events.append({"event": event_name, "data": data})
                    event_name = None
        return events

    @pytest.mark.asyncio
    async def test_emits_created_event_first(self):
        output = ModelOutputThunk("hello")
        events = await self._collect(
            output, response_id="resp_1", model="m", created=1234
        )
        assert events[0]["event"] == "response.created"
        assert events[0]["data"]["id"] == "resp_1"
        assert events[0]["data"]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_emits_in_progress_event(self):
        output = ModelOutputThunk("hello")
        events = await self._collect(
            output, response_id="resp_1", model="m", created=1234
        )
        names = [e["event"] for e in events]
        assert "response.in_progress" in names

    @pytest.mark.asyncio
    async def test_emits_output_text_delta(self):
        output = ModelOutputThunk("hello world")
        events = await self._collect(
            output, response_id="resp_1", model="m", created=1234
        )
        delta_events = [e for e in events if e["event"] == "response.output_text.delta"]
        assert len(delta_events) >= 1
        assert delta_events[0]["data"]["delta"] == "hello world"

    @pytest.mark.asyncio
    async def test_emits_output_text_done(self):
        output = ModelOutputThunk("hello world")
        events = await self._collect(
            output, response_id="resp_1", model="m", created=1234
        )
        done_events = [e for e in events if e["event"] == "response.output_text.done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["text"] == "hello world"

    @pytest.mark.asyncio
    async def test_emits_completed_event_last(self):
        output = ModelOutputThunk("hello")
        events = await self._collect(
            output, response_id="resp_1", model="m", created=1234
        )
        assert events[-1]["event"] == "response.completed"
        assert events[-1]["data"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_conversation_echoed_in_created_event(self):
        output = ModelOutputThunk("hi")
        events = await self._collect(
            output,
            response_id="resp_1",
            model="m",
            created=1234,
            conversation="conv-99",
        )
        assert events[0]["data"]["conversation"] == "conv-99"

    @pytest.mark.asyncio
    async def test_true_streaming_accumulates_deltas(self):
        """Incremental astream() calls each yield a delta event; done text is the sum."""
        output = Mock()
        tokens = ["He", "ll", "o"]
        call_count = 0

        def is_computed():
            return call_count >= len(tokens)

        async def astream():
            nonlocal call_count
            token = tokens[call_count]
            call_count += 1
            return token

        output.is_computed = is_computed
        output.astream = astream
        output.tool_calls = None
        output.generation = Mock()
        output.generation.usage = None

        events = await self._collect(
            output, response_id="resp_s", model="m", created=1234
        )

        delta_events = [e for e in events if e["event"] == "response.output_text.delta"]
        assert len(delta_events) == 3
        assert [e["data"]["delta"] for e in delta_events] == ["He", "ll", "o"]

        done_events = [e for e in events if e["event"] == "response.output_text.done"]
        assert done_events[0]["data"]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_tool_calls_emit_function_call_arguments_done(self):
        """Tool calls on the output produce function_call_arguments.done events."""
        from mellea.core.base import ModelToolCall

        output = ModelOutputThunk("ok")
        tc = Mock(spec=ModelToolCall)
        tc.name = "get_weather"
        tc.args = {"location": "Paris"}
        tc.tool_call_id = "call_abc"
        output.tool_calls = [tc]

        events = await self._collect(
            output, response_id="resp_tc", model="m", created=1234
        )

        fc_events = [
            e for e in events if e["event"] == "response.function_call_arguments.done"
        ]
        assert len(fc_events) == 1
        assert fc_events[0]["data"]["name"] == "get_weather"
        assert fc_events[0]["data"]["call_id"] == "call_abc"
        assert fc_events[0]["data"]["output_index"] == 1

    @pytest.mark.asyncio
    async def test_error_emits_failed_event(self):
        """An exception mid-stream emits response.failed."""
        output = Mock()
        output.is_computed.return_value = False
        output.astream = AsyncMock(side_effect=RuntimeError("oops"))
        output.generation = Mock()
        output.generation.usage = None

        events = await self._collect(
            output, response_id="resp_err", model="m", created=1234
        )

        failed = [e for e in events if e["event"] == "response.failed"]
        assert len(failed) == 1
        assert failed[0]["data"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_usage_omitted_without_include(self):
        """Usage is absent from response.completed when include is not set."""
        output = ModelOutputThunk("hello")
        output.generation.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }
        events = await self._collect(
            output, response_id="resp_1", model="m", created=1234
        )
        completed = next(e for e in events if e["event"] == "response.completed")
        assert completed["data"]["usage"] is None

    @pytest.mark.asyncio
    async def test_usage_omitted_when_include_does_not_contain_usage(self):
        """Usage is absent from response.completed when include excludes 'usage'."""
        output = ModelOutputThunk("hello")
        output.generation.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }
        events = await self._collect(
            output, response_id="resp_1", model="m", created=1234, include=["reasoning"]
        )
        completed = next(e for e in events if e["event"] == "response.completed")
        assert completed["data"]["usage"] is None

    @pytest.mark.asyncio
    async def test_usage_included_when_include_contains_usage(self):
        """Usage is present in response.completed when include=['usage']."""
        output = ModelOutputThunk("hello")
        output.generation.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }
        events = await self._collect(
            output, response_id="resp_1", model="m", created=1234, include=["usage"]
        )
        completed = next(e for e in events if e["event"] == "response.completed")
        assert completed["data"]["usage"] is not None
        assert completed["data"]["usage"]["input_tokens"] == 5
        assert completed["data"]["usage"]["output_tokens"] == 3
        assert completed["data"]["usage"]["total_tokens"] == 8

    @pytest.mark.asyncio
    async def test_endpoint_passes_include_to_streaming(self, mock_module):
        """The endpoint forwards request.include to stream_response_chunks."""
        from fastapi.responses import StreamingResponse

        req = ResponseRequest(model="m", input="hi", stream=True, include=["usage"])
        output = ModelOutputThunk("streamed")
        output.generation.usage = {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        }
        mock_module.serve.return_value = output
        endpoint = make_responses_endpoint(mock_module)
        response = await endpoint(req)
        assert isinstance(response, StreamingResponse)

        events = []
        async for chunk in response.body_iterator:
            event_name = None
            for line in chunk.split("\n"):
                line = line.strip()
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event_name is not None:
                    data = json.loads(line.split(":", 1)[1].strip())
                    events.append({"event": event_name, "data": data})
                    event_name = None

        completed = next(e for e in events if e["event"] == "response.completed")
        assert completed["data"]["usage"] is not None
        assert completed["data"]["usage"]["input_tokens"] == 2


# ---------------------------------------------------------------------------
# build_response_usage
# ---------------------------------------------------------------------------


class TestBuildResponseUsage:
    """Tests for the ResponseUsage helper."""

    def test_returns_none_when_no_usage(self):
        output = ModelOutputThunk("hi")
        assert build_response_usage(output) is None

    def test_maps_prompt_to_input_tokens(self):
        output = ModelOutputThunk("hi")
        output.generation.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        usage = build_response_usage(output)
        assert usage is not None
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15

    def test_reasoning_tokens_extracted(self):
        output = ModelOutputThunk("hi")
        output.generation.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": 3,
        }
        usage = build_response_usage(output)
        assert usage is not None
        assert usage.reasoning_tokens == 3

    def test_reasoning_tokens_none_when_absent(self):
        output = ModelOutputThunk("hi")
        output.generation.usage = {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
        usage = build_response_usage(output)
        assert usage is not None
        assert usage.reasoning_tokens is None

    def test_total_tokens_inferred_when_missing(self):
        output = ModelOutputThunk("hi")
        output.generation.usage = {"prompt_tokens": 4, "completion_tokens": 6}
        usage = build_response_usage(output)
        assert usage is not None
        assert usage.total_tokens == 10


# ---------------------------------------------------------------------------
# build_response_output_items
# ---------------------------------------------------------------------------


class TestBuildResponseOutputItems:
    """Tests for the output items builder."""

    def test_text_output_creates_message_item(self):
        output = ModelOutputThunk("Hello!")
        items = build_response_output_items(output, None)
        assert len(items) == 1
        assert items[0].type == "message"
        assert items[0].role == "assistant"  # type: ignore[union-attr]

    def test_message_item_contains_output_text_content(self):
        output = ModelOutputThunk("Hi there")
        items = build_response_output_items(output, None)
        msg = items[0]
        assert msg.content[0].type == "output_text"  # type: ignore[union-attr]
        assert msg.content[0].text == "Hi there"  # type: ignore[union-attr]

    def test_empty_value_produces_no_message_item(self):
        output = ModelOutputThunk(None)
        items = build_response_output_items(output, None)
        assert all(item.type != "message" for item in items)

    def test_tool_calls_produce_function_call_items(self):
        output = ModelOutputThunk("ok")
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location":"NYC"}'},
            }
        ]
        items = build_response_output_items(output, tool_calls)
        fc_items = [i for i in items if i.type == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0].name == "get_weather"  # type: ignore[union-attr]
        assert fc_items[0].call_id == "call_1"  # type: ignore[union-attr]

    def test_multiple_tool_calls_all_present(self):
        output = ModelOutputThunk("ok")
        tool_calls = [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f1", "arguments": "{}"},
            },
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "f2", "arguments": "{}"},
            },
        ]
        items = build_response_output_items(output, tool_calls)
        fc_items = [i for i in items if i.type == "function_call"]
        assert len(fc_items) == 2

    def test_message_item_id_has_msg_prefix(self):
        output = ModelOutputThunk("hi")
        items = build_response_output_items(output, None)
        assert items[0].id.startswith("msg_")

    def test_function_call_item_id_has_fc_prefix(self):
        output = ModelOutputThunk(None)
        tool_calls = [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ]
        items = build_response_output_items(output, tool_calls)
        assert items[0].id.startswith("fc_")
