"""A simple app that runs an OpenAI compatible server wrapped around a M program."""

import asyncio
import importlib.util
import inspect
import os
import sys
import time
import uuid

import typer
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mellea.backends.model_options import ModelOption

from .models import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionRequest,
    Choice,
    CompletionUsage,
    OpenAIError,
    OpenAIErrorResponse,
)

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
        # Not-yet-implemented OpenAI parameters (silently ignored)
        "stream",  # Streaming responses - not yet implemented
        "stop",  # Stop sequences - not yet implemented
        "top_p",  # Nucleus sampling - not yet implemented
        "presence_penalty",  # Presence penalty - not yet implemented
        "frequency_penalty",  # Frequency penalty - not yet implemented
        "logit_bias",  # Logit bias - not yet implemented
        "response_format",  # Response format (json_object) - not yet implemented
        "functions",  # Legacy function calling - not yet implemented
        "function_call",  # Legacy function calling - not yet implemented
        "tools",  # Tool calling - not yet implemented
        "tool_choice",  # Tool choice - not yet implemented
    }
    openai_to_model_option = {
        "temperature": ModelOption.TEMPERATURE,
        "max_tokens": ModelOption.MAX_NEW_TOKENS,
        "seed": ModelOption.SEED,
    }

    filtered_options = {
        key: value
        for key, value in request.model_dump(exclude_none=True).items()
        if key not in excluded_fields
    }
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

            # Extract usage information from the ModelOutputThunk if available
            usage = None
            if hasattr(output, "usage") and output.usage is not None:
                prompt_tokens = output.usage.get("prompt_tokens", 0)
                completion_tokens = output.usage.get("completion_tokens", 0)
                # Calculate total_tokens if not provided
                total_tokens = output.usage.get(
                    "total_tokens", prompt_tokens + completion_tokens
                )
                usage = CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            # system_fingerprint represents backend config hash, not model name
            # The model name is already in response.model (line 73)
            # Leave as None since we don't track backend config fingerprints yet
            system_fingerprint = None

            return ChatCompletion(
                id=completion_id,
                model=request.model,
                created=created_timestamp,
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(
                            content=output.value, role="assistant"
                        ),
                        finish_reason="stop",
                    )
                ],
                object="chat.completion",  # type: ignore
                system_fingerprint=system_fingerprint,
                usage=usage,
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


def serve(
    script_path: str = typer.Argument(
        default="docs/examples/m_serve/example.py",
        help="Path to the Python script to import and serve",
    ),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8080, help="Port to bind to"),
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
