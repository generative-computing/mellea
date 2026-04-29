"""A simple app that runs an OpenAI compatible server wrapped around a M program."""

import asyncio
import importlib.util
import inspect
import os
import sys
import time
import uuid
from typing import Literal

try:
    import typer
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as e:
    raise ImportError(
        "The 'm serve' command requires extra dependencies. "
        'Please install them with: pip install "mellea[server]"'
    ) from e

from mellea.backends.model_options import ModelOption
from mellea.helpers.openai_compatible_helpers import (
    build_completion_usage,
    build_tool_calls,
)

from .models import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionRequest,
    Choice,
    OpenAIError,
    OpenAIErrorResponse,
    ToolCallFunction,
)
from .streaming import stream_chat_completion_chunks

app = FastAPI(
    title="M serve OpenAI API Compatible Server",
    description="M programs that run as a simple OpenAI API-compatible server",
    version="0.1.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Convert FastAPI validation errors to OpenAI-compatible format.

    FastAPI returns 422 with a 'detail' array by default. OpenAI API uses
    400 with an 'error' object containing message, type, and param fields.
    """
    # Extract the first validation error
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        # Get the field name from the location tuple (e.g., ('body', 'n') -> 'n')
        param = first_error["loc"][-1] if first_error["loc"] else None
        message = first_error["msg"]
    else:
        param = None
        message = "Invalid request parameters"

    return create_openai_error_response(
        status_code=400,
        message=message,
        error_type="invalid_request_error",
        param=str(param) if param else None,
    )


def load_module_from_path(path: str):
    """Load the module with M program in it."""
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def create_openai_error_response(
    status_code: int, message: str, error_type: str, param: str | None = None
) -> JSONResponse:
    """Create an OpenAI-compatible error response."""
    error_response = OpenAIErrorResponse(
        error=OpenAIError(message=message, type=error_type, param=param)
    )
    return JSONResponse(
        status_code=status_code, content=error_response.model_dump(mode="json")
    )


def _build_model_options(request: ChatCompletionRequest) -> dict:
    """Build model_options dict from OpenAI-compatible request parameters."""
    excluded_fields = {
        # Request structure fields (handled separately)
        "messages",  # Chat messages - passed separately to serve()
        "requirements",  # Mellea requirements - passed separately to serve()
        # Routing/metadata fields (not generation parameters)
        "model",  # Model identifier - used for routing, not generation
        "n",  # Number of completions - not supported in Mellea's model_options
        "user",  # User tracking ID - metadata, not a generation parameter
        "extra",  # Pydantic's extra fields dict - unused (see model_config)
        "stream_options",  # Streaming options - handled separately in streaming response
        # Not-yet-implemented OpenAI parameters (silently ignored)
        "stop",  # Stop sequences - not yet implemented
        "top_p",  # Nucleus sampling - not yet implemented
        "presence_penalty",  # Presence penalty - not yet implemented
        "frequency_penalty",  # Frequency penalty - not yet implemented
        "logit_bias",  # Logit bias - not yet implemented
        "response_format",  # Response format (json_object) - not yet implemented
        "functions",  # Legacy function calling - not yet implemented
        "function_call",  # Legacy function calling - not yet implemented
    }
    openai_to_model_option = {
        "temperature": ModelOption.TEMPERATURE,
        "max_tokens": ModelOption.MAX_NEW_TOKENS,
        "seed": ModelOption.SEED,
        "stream": ModelOption.STREAM,
        "tools": ModelOption.TOOLS,
        "tool_choice": ModelOption.TOOL_CHOICE,
    }

    # Get all non-None fields
    filtered_options = {
        key: value
        for key, value in request.model_dump(exclude_none=True).items()
        if key not in excluded_fields
    }

    # Special handling for stream: only include if True (don't forward False)
    if "stream" in filtered_options and not filtered_options["stream"]:
        del filtered_options["stream"]

    return ModelOption.replace_keys(filtered_options, openai_to_model_option)


def make_chat_endpoint(module):
    """Makes a chat endpoint using a custom module."""

    async def endpoint(request: ChatCompletionRequest):
        try:
            # Validate that n=1 (we don't support multiple completions)
            if request.n is not None and request.n > 1:
                return create_openai_error_response(
                    status_code=400,
                    message=f"Multiple completions (n={request.n}) are not supported. Please set n=1 or omit the parameter.",
                    error_type="invalid_request_error",
                    param="n",
                )

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created_timestamp = int(time.time())

            model_options = _build_model_options(request)

            # Detect if serve is async or sync and handle accordingly
            if inspect.iscoroutinefunction(module.serve):
                # It's async, await it directly
                output = await module.serve(
                    input=request.messages,
                    requirements=request.requirements,
                    model_options=model_options,
                )
            else:
                # It's sync, run in thread pool to avoid blocking event loop
                output = await asyncio.to_thread(
                    module.serve,
                    input=request.messages,
                    requirements=request.requirements,
                    model_options=model_options,
                )

            # Leave as None since we don't track backend config fingerprints yet
            system_fingerprint = None

            # Handle streaming response
            if request.stream:
                return StreamingResponse(
                    stream_chat_completion_chunks(
                        output=output,
                        completion_id=completion_id,
                        model=request.model,
                        created=created_timestamp,
                        stream_options=request.stream_options,
                        system_fingerprint=system_fingerprint,
                    ),
                    media_type="text/event-stream",
                )

            # Extract tool calls from the ModelOutputThunk if available
            tool_calls_list = build_tool_calls(output)
            tool_calls = (
                [
                    ChatCompletionMessageToolCall(
                        id=tc["id"],
                        type=tc["type"],
                        function=ToolCallFunction(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in tool_calls_list
                ]
                if tool_calls_list
                else None
            )

            # Determine finish_reason based on tool calls
            finish_reason: (
                Literal[
                    "stop", "length", "content_filter", "tool_calls", "function_call"
                ]
                | None
            ) = "tool_calls" if tool_calls else "stop"

            return ChatCompletion(
                id=completion_id,
                model=request.model,
                created=created_timestamp,
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(
                            content=output.value,
                            role="assistant",
                            tool_calls=tool_calls,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                object="chat.completion",  # type: ignore
                system_fingerprint=system_fingerprint,
                usage=build_completion_usage(output),
            )  # type: ignore
        except ValueError as e:
            # Handle validation errors or invalid input
            return create_openai_error_response(
                status_code=400,
                message=f"Invalid request: {e!s}",
                error_type="invalid_request_error",
            )
        except Exception as e:
            # Catch-all for any unexpected errors (including AttributeError)
            return create_openai_error_response(
                status_code=500,
                message=f"Internal server error: {e!s}",
                error_type="server_error",
            )

    endpoint.__name__ = f"chat_{module.__name__}_endpoint"
    return endpoint


def run_server(
    script_path: str = "docs/examples/m_serve/example.py",
    host: str = "0.0.0.0",
    port: int = 8080,
):
    """Serve a FastAPI endpoint for a given script."""
    module = load_module_from_path(script_path)
    route_path = "/v1/chat/completions"

    app.add_api_route(
        route_path,
        make_chat_endpoint(module),
        methods=["POST"],
        response_model=ChatCompletion | OpenAIErrorResponse,
    )
    typer.echo(f"Serving {route_path} at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
