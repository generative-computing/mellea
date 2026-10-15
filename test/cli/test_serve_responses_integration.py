# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-level integration tests for the /v1/responses endpoint.

Drives the endpoint through the real OpenAI SDK with a FastAPI TestClient
as the SDK's http_client (no network): request parsing, the SSE event
sequence, and streamed-turn storage are exercised the way SDK clients
consume them.
"""

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import openai
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel

import cli.serve.app as app_module
from cli.serve.app import (
    _lifespan,
    _response_store,
    get_response,
    make_responses_endpoint,
    validation_exception_handler,
)
from cli.serve.models import OpenAIErrorResponse, Response
from mellea.core.base import AbstractMelleaTool, ModelOutputThunk, ModelToolCall

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

PERSON_SCHEMA: dict = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


class MockWeatherTool(AbstractMelleaTool):
    """Mock weather tool for the tool round-trip test."""

    name = "get_weather"

    def run(self, location: str, units: str = "celsius") -> str:
        """Return canned weather."""
        return f"Weather in {location} is sunny and 22°C"

    @property
    def as_json_tool(self) -> dict[str, Any]:
        """Return JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        }


def _build_app(module: Any) -> FastAPI:
    """Build a fresh app wired like the real server.

    Never the global app from cli.serve.app — add_api_route would stack
    duplicate routes. The real _lifespan is attached so every test exercises
    it, and the GET route is registered explicitly (on the real server it is
    only on the global app).
    """
    app = FastAPI(lifespan=_lifespan)
    # The handler's narrow (Request, RequestValidationError) signature is
    # fine at runtime; Starlette's declared handler type is wider.
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,  # type: ignore[arg-type]
    )
    app.add_api_route(
        "/v1/responses",
        make_responses_endpoint(module),
        methods=["POST"],
        response_model=Response | OpenAIErrorResponse,
    )
    app.add_api_route(
        "/v1/responses/{response_id}",
        get_response,
        methods=["GET"],
        response_model=Response | OpenAIErrorResponse,
    )
    return app


@contextmanager
def _sdk_client(module: Any) -> Iterator[openai.OpenAI]:
    """OpenAI SDK client whose http_client is a TestClient of the app (no network).

    The TestClient is entered as a context manager so the app's lifespan
    (eviction task startup/shutdown) runs for real on every test.
    """
    with TestClient(_build_app(module)) as tc:
        yield openai.OpenAI(api_key="na", base_url="http://test/v1", http_client=tc)


@pytest.fixture
def mock_module():
    """Mock module with a synchronous serve function."""
    module = Mock()
    module.__name__ = "test_responses_integration_module"
    return module


@pytest.fixture(autouse=True)
def _clear_store():
    """Keep the module-level response store isolated between tests."""
    _response_store.clear()
    yield
    _response_store.clear()


class TestResponsesSdkIntegration:
    """Full SDK → HTTP → endpoint round trips."""

    def test_create_non_streaming(self, mock_module):
        """client.responses.create() parses the response the SDK way."""
        mock_module.serve.return_value = ModelOutputThunk("Hello from serve")

        with _sdk_client(mock_module) as client:
            response = client.responses.create(model="test-model", input="Hi")

        assert response.output_text == "Hello from serve"
        assert response.id.startswith("resp_")
        assert response.status == "completed"
        assert isinstance(response.usage.total_tokens, int)
        # Required spec fields populated by the usage helper (zeros when unknown)
        assert isinstance(response.usage.output_tokens_details.reasoning_tokens, int)

    def test_stream_helper(self, mock_module):
        """client.responses.stream() consumes the SSE event sequence."""
        mock_module.serve.return_value = ModelOutputThunk("Count down from ten")

        deltas: list[str] = []
        with _sdk_client(mock_module) as client:
            with client.responses.stream(model="test-model", input="Hi") as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        deltas.append(event.delta)
                final = stream.get_final_response()

        assert "".join(deltas) == "Count down from ten"
        assert final.output_text == "Count down from ten"
        assert final.status == "completed"

    def test_text_format_structured_output(self, mock_module):
        """The SDK's text={"format": ...} shape reaches serve() as a model."""
        received: dict[str, Any] = {}

        def capture_format(input, requirements=None, model_options=None, format=None):
            received["format"] = format
            return ModelOutputThunk('{"name": "Alice", "age": 30}')

        mock_module.serve = capture_format

        with _sdk_client(mock_module) as client:
            response = client.responses.create(
                model="test-model",
                input="Generate a person",
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "Person",
                        "schema": PERSON_SCHEMA,
                        "strict": True,
                    }
                },
            )

        assert response.output_text == '{"name": "Alice", "age": 30}'
        assert received["format"] is not None
        assert issubclass(received["format"], BaseModel)
        assert "name" in received["format"].model_fields
        assert "age" in received["format"].model_fields

    def test_tool_round_trip(self, mock_module):
        """Flat SDK tools, function_call output items, function_call_output back."""
        serve_inputs: list[list] = []

        def serve(input, requirements=None, model_options=None):
            serve_inputs.append(list(input))
            if len(serve_inputs) == 1:
                thunk = ModelOutputThunk("Let me check the weather.")
                thunk.tool_calls = [
                    ModelToolCall(
                        name="get_weather",
                        func=MockWeatherTool(),
                        args={"location": "Paris"},
                        tool_call_id="call_123",
                    )
                ]
                return thunk
            return ModelOutputThunk("It is sunny in Paris.")

        mock_module.serve = serve

        with _sdk_client(mock_module) as client:
            first = client.responses.create(
                model="test-model",
                input="What's the weather in Paris?",
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get the weather in a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    }
                ],
                tool_choice="auto",
            )
            assert first.status == "completed"

            function_calls = [
                item for item in first.output if item.type == "function_call"
            ]
            assert len(function_calls) == 1
            fc = function_calls[0]
            assert fc.name == "get_weather"
            assert json.loads(fc.arguments) == {"location": "Paris"}
            assert fc.call_id == "call_123"

            second = client.responses.create(
                model="test-model",
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": "Sunny, 25C",
                    }
                ],
                previous_response_id=first.id,
            )
            assert second.output_text == "It is sunny in Paris."

        # serve()'s second call received the tool result as a tool message.
        second_input = serve_inputs[1]
        tool_messages = [m for m in second_input if getattr(m, "role", None) == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "call_123"
        assert tool_messages[0].content == "Sunny, 25C"

    def test_previous_response_id_after_streamed_turn(self, mock_module):
        """A streamed turn is stored, retrievable, and chainable."""
        serve_inputs: list[list] = []

        def serve(input, requirements=None, model_options=None):
            serve_inputs.append(list(input))
            return ModelOutputThunk(f"reply {len(serve_inputs)}")

        mock_module.serve = serve

        with _sdk_client(mock_module) as client:
            with client.responses.stream(
                model="test-model", input="first question"
            ) as stream:
                first = stream.get_final_response()

            # The streamed turn was persisted: retrievable via GET...
            retrieved = client.responses.retrieve(first.id)
            assert retrieved.id == first.id
            assert retrieved.output_text == "reply 1"

            # ...and chainable via previous_response_id.
            second = client.responses.create(
                model="test-model",
                input="second question",
                previous_response_id=first.id,
            )
            assert second.output_text == "reply 2"

        # serve()'s second call received the first turn's history.
        history_text = " ".join(str(getattr(m, "content", "")) for m in serve_inputs[1])
        assert "first question" in history_text
        assert "reply 1" in history_text


class TestLifespan:
    """The app lifespan schedules and cancels the eviction task."""

    def test_lifespan_schedules_eviction_task(self, mock_module):
        """_lifespan starts _evict_expired_responses and cancels it on shutdown."""
        started: list[bool] = []

        async def fake_evict() -> None:
            started.append(True)
            await asyncio.Event().wait()  # block until cancelled

        mock_module.serve.return_value = ModelOutputThunk("ok")

        with patch.object(app_module, "_evict_expired_responses", fake_evict):
            with TestClient(_build_app(mock_module)) as tc:
                # A request cycles the event loop so the scheduled task starts.
                resp = tc.post("/v1/responses", json={"model": "m", "input": "hi"})
                assert resp.status_code == 200

        assert started == [True]
