"""A simple app that runs an OpenAI compatible server wrapped around a M program."""

import importlib.util
import os
import sys
import time
import uuid

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

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


def make_chat_endpoint(module):
    """Makes a chat endpoint using a custom module."""

    async def endpoint(request: ChatCompletionRequest):
        try:
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created_timestamp = int(time.time())

            output = module.serve(
                input=request.messages,
                requirements=request.requirements,
                model_options={
                    k: v
                    for k, v in request.model_dump().items()
                    if k not in ["messages", "requirements"]
                },
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
        except AttributeError as e:
            # Handle missing 'value' attribute or other attribute errors
            return create_openai_error_response(
                status_code=500,
                message=f"Internal server error: {e!s}",
                error_type="server_error",
            )
        except ValueError as e:
            # Handle validation errors or invalid input
            return create_openai_error_response(
                status_code=400,
                message=f"Invalid request: {e!s}",
                error_type="invalid_request_error",
            )
        except Exception as e:
            # Catch-all for any other unexpected errors
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
        response_model=None,  # Allow both ChatCompletion and error responses
    )
    typer.echo(f"Serving {route_path} at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
