# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simple app that runs an OpenAI compatible server wrapped around a M program."""

import asyncio
import importlib.util
import inspect
import os
import secrets
import string
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Literal, cast

try:
    from contextlib import asynccontextmanager

    import typer
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "The 'm serve' command requires extra dependencies. "
        'Please install them with: pip install "mellea[server]"'
    ) from e

from mellea.backends.model_options import ModelOption
from mellea.core import MelleaLogger
from mellea.helpers.openai_compatible_helpers import (
    build_completion_usage,
    build_response_output_items,
    build_response_usage,
    build_tool_calls,
)

from .models import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionRequest,
    Choice,
    InputContent,
    JsonSchemaFormat,
    OpenAIError,
    OpenAIErrorResponse,
    Response,
    ResponseInputItem,
    ResponseRequest,
    ResponseUsage,
)
from .schema_converter import json_schema_to_pydantic
from .streaming import stream_chat_completion_chunks, stream_response_chunks
from .utils import extract_finish_reason

logger = MelleaLogger.get_logger()

_BASE62 = string.ascii_letters + string.digits


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Start background tasks on server startup."""
    task = asyncio.create_task(_evict_expired_responses())
    yield
    task.cancel()


app = FastAPI(
    title="M serve OpenAI API Compatible Server",
    description="M programs that run as a simple OpenAI API-compatible server",
    version="0.1.0",
    lifespan=_lifespan,
)

# ---------------------------------------------------------------------------
# In-memory response store
#
# Keyed by response_id (resp_…).  Each entry holds the completed Response
# object plus the full message history so that a future request can pass
# previous_response_id to continue the conversation without re-sending
# history.  Entries are evicted after _response_ttl_seconds by a background
# sweep task started in run_server().
# ---------------------------------------------------------------------------

_response_ttl_seconds: int = 1800  # 30 minutes, overridden by run_server()


@dataclass
class _StoredResponse:
    response: Response
    # Ordered list of ChatMessage-compatible dicts representing the full
    # conversation up to and including this response, in a format both
    # endpoints can consume without importing Responses API types.
    history: list[dict[str, Any]]
    stored_at: float = field(default_factory=time.time)


_response_store: dict[str, _StoredResponse] = {}


async def _evict_expired_responses() -> None:
    """Background task: evict store entries older than _response_ttl_seconds."""
    while True:
        await asyncio.sleep(60)  # sweep every minute
        cutoff = time.time() - _response_ttl_seconds
        expired = [
            rid for rid, entry in _response_store.items() if entry.stored_at < cutoff
        ]
        for rid in expired:
            del _response_store[rid]
        if expired:
            logger.debug("Evicted %d expired response(s) from store", len(expired))


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic liveness check endpoint.

    Returns a 200 OK status to signal that the Python process is alive and responding.

    Returns:
        dict: A dictionary with status "pass".
    """
    return {"status": "pass"}


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
        "response_format",  # Response format - handled separately
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


def _build_client_options(request: ChatCompletionRequest | ResponseRequest) -> dict:
    """Return the full raw client request as a plain dict.

    Passed to serve() as client_options when the function declares that
    parameter, giving it access to every field the client sent (including
    routing and metadata fields like model, user, and n) while ensuring those
    specific named routing fields are excluded from model_options. Note that
    arbitrary extra fields are forwarded to model_options as potential
    backend-specific generation parameters.
    """
    return request.model_dump(exclude_none=True)


def _build_model_options_from_response_request(request: ResponseRequest) -> dict:
    """Build model_options dict from Responses API request parameters."""
    excluded_fields = {
        "input",
        "instructions",
        "model",
        "stream",
        "store",
        "conversation",
        "previous_response_id",
        "background",
        "include",
        "parallel_tool_calls",
        "context_management",
        "prompt_cache_options",
        "format",  # Handled separately for structured outputs
        "extra",
        # Not-yet-implemented
        "top_p",
    }
    openai_to_model_option = {
        "max_output_tokens": ModelOption.MAX_NEW_TOKENS,
        "temperature": ModelOption.TEMPERATURE,
        "tools": ModelOption.TOOLS,
        "tool_choice": ModelOption.TOOL_CHOICE,
    }

    filtered_options = {
        key: value
        for key, value in request.model_dump(exclude_none=True).items()
        if key not in excluded_fields
    }

    result = ModelOption.replace_keys(filtered_options, openai_to_model_option)

    # Only "function" tool types are supported. Native tool types (web_search,
    # file_search, mcp, code_interpreter) require OpenAI server-side execution
    # and are not implemented. Reject them here so callers get a clear 400
    # instead of an internal AssertionError deep in the backend.
    _UNSUPPORTED_TOOL_TYPES = {"web_search", "file_search", "mcp", "code_interpreter"}
    if ModelOption.TOOLS in result:
        unsupported = [
            t["type"]
            for t in result[ModelOption.TOOLS]
            if t.get("type") in _UNSUPPORTED_TOOL_TYPES
        ]
        if unsupported:
            raise ValueError(
                f"Tool type(s) {unsupported} are not supported by m serve. "
                "Only 'function' tools are supported. Native tool types "
                "(web_search, file_search, mcp, code_interpreter) require "
                "server-side execution that is not yet implemented."
            )

    return result


def _convert_input_content_to_message_content(
    content: str | list[InputContent] | None,
) -> str | list | None:
    """Convert Responses API InputContent list to ChatMessage content format.

    Handles both string content (backward compatible) and list of content
    blocks (multimodal: text, images, files).

    Args:
        content: Either a string (text-only), a list of InputContent blocks,
            or None.

    Returns:
        - str: If input is a string
        - list[MessageContent]: If input is a list of content blocks
        - None: If input is None

    Raises:
        ValueError: If an unsupported content type is encountered.
    """
    from mellea.serve.models import ImageUrlContent, TextContent

    if content is None or isinstance(content, str):
        return content

    # Convert list of InputContent to list of MessageContent
    message_content: list = []
    for block in content:
        if block.type == "input_text" and block.text is not None:
            message_content.append(TextContent(type="text", text=block.text))
        elif block.type == "input_image" and block.image_url is not None:
            message_content.append(
                ImageUrlContent(type="image_url", image_url={"url": block.image_url})
            )
        elif block.type == "input_file":
            # File content is not yet supported - skip with a warning
            logger.warning(
                "File content (type='input_file') is not yet supported in multimodal "
                "messages. This content block will be ignored."
            )
        else:
            logger.warning(
                f"Unsupported content block type: {block.type!r}. "
                "This content block will be ignored."
            )

    return message_content if message_content else None


def _convert_response_input_to_messages(
    input_data: str | list[ResponseInputItem],
    instructions: str | list[ResponseInputItem] | None,
) -> list:
    """Convert Responses API input + instructions to a ChatMessage list.

    Handles string shorthand, message arrays, and developer/system role
    mapping. The Responses API uses ``developer`` where Chat Completions uses
    ``system``; both are mapped to ``system`` so existing backends receive a
    role they understand.

    Supports multimodal content: text, images (via image_url), and files
    (not yet supported). Content blocks are converted from Responses API
    format (InputContent) to ChatMessage format (MessageContent).
    """
    from mellea.serve.models import ChatMessage

    def _role(r: str) -> Literal["system", "user", "assistant", "tool", "function"]:
        if r in ("developer", "system"):
            return "system"
        if r in ("user", "assistant", "tool", "function"):
            return cast(Literal["system", "user", "assistant", "tool", "function"], r)
        raise ValueError(f"Unexpected role: {r!r}")

    messages: list[ChatMessage] = []

    if instructions:
        if isinstance(instructions, str):
            messages.append(ChatMessage(role="system", content=instructions))
        else:
            for item in instructions:
                content = _convert_input_content_to_message_content(item.content)
                messages.append(ChatMessage(role=_role(item.role), content=content))

    if isinstance(input_data, str):
        messages.append(ChatMessage(role="user", content=input_data))
    else:
        for item in input_data:
            content = _convert_input_content_to_message_content(item.content)
            messages.append(ChatMessage(role=_role(item.role), content=content))

    return messages


def _build_history_from_messages(
    messages: list, assistant_text: str
) -> list[dict[str, Any]]:
    """Serialize a message list + assistant reply into a plain dict history.

    The history is stored in a format that both the responses and chat/completions
    endpoints can consume: a list of ``{"role": str, "content": str}`` dicts,
    matching the ChatMessage wire format.
    """
    history: list[dict[str, Any]] = [
        {"role": m.role, "content": m.content} for m in messages
    ]
    history.append({"role": "assistant", "content": assistant_text})
    return history


def make_responses_endpoint(module):
    """Makes a /v1/responses endpoint using a custom module.

    Mirrors make_chat_endpoint() with Responses API-specific adaptations:
    input conversion, semantic streaming, and Response output format.

    Supports multi-turn sessions via ``previous_response_id``: when set, the
    server looks up the stored history from that response and prepends it to
    the current input, so the client only needs to send the new turn.
    Completed responses are stored in the in-memory ``_response_store`` when
    ``store=True`` (the default) and expire after ``_response_ttl_seconds``.
    """
    serve_sig = inspect.signature(module.serve)
    accepts_format = "format" in serve_sig.parameters
    accepts_client_options = "client_options" in serve_sig.parameters
    is_async = inspect.iscoroutinefunction(module.serve)

    async def endpoint(request: ResponseRequest):
        try:
            if request.background:
                return create_openai_error_response(
                    status_code=400,
                    message="Background processing is not yet supported.",
                    error_type="invalid_request_error",
                    param="background",
                )

            response_id = "resp_" + "".join(secrets.choice(_BASE62) for _ in range(24))
            created_timestamp = int(time.time())

            messages = _convert_response_input_to_messages(
                request.input, request.instructions
            )

            # Prepend history from a previous response if the client is
            # continuing an existing conversation.
            if request.previous_response_id:
                stored = _response_store.get(request.previous_response_id)
                if stored is None:
                    return create_openai_error_response(
                        status_code=404,
                        message=f"Response '{request.previous_response_id}' not found or has expired.",
                        error_type="invalid_request_error",
                        param="previous_response_id",
                    )
                from mellea.serve.models import ChatMessage

                prior_messages = [
                    ChatMessage(role=m["role"], content=m["content"])
                    for m in stored.history
                ]
                messages = prior_messages + messages

            model_options = _build_model_options_from_response_request(request)

            # Handle format (structured outputs)
            format_model: type[BaseModel] | None = None
            if request.format is not None:
                if request.format.type == "json_schema":
                    # json_schema presence is validated by ResponseFormat.model_validator
                    json_schema = cast(JsonSchemaFormat, request.format.json_schema)
                    try:
                        format_model = json_schema_to_pydantic(
                            json_schema.schema_, json_schema.name
                        )
                    except (ValueError, TypeError, RecursionError) as e:
                        message = (
                            "Invalid JSON schema: recursive $ref is not supported"
                            if isinstance(e, RecursionError)
                            else f"Invalid JSON schema: {e!s}"
                        )
                        return create_openai_error_response(
                            status_code=400,
                            message=message,
                            error_type="invalid_request_error",
                            param="format.json_schema.schema",
                        )
                # For "json_object" and "text", format_model remains None
                # Note: "json_object" mode is not yet implemented - the backend
                # receives no signal to produce JSON output (same as "text" mode)

            serve_kwargs: dict[str, Any] = {
                "input": messages,
                "requirements": None,
                "model_options": model_options,
            }
            if accepts_format:
                serve_kwargs["format"] = format_model
            if accepts_client_options:
                serve_kwargs["client_options"] = _build_client_options(request)

            if is_async:
                output = await module.serve(**serve_kwargs)
            else:
                output = await asyncio.to_thread(module.serve, **serve_kwargs)

            storing = request.store is not False

            if request.stream:
                return StreamingResponse(
                    stream_response_chunks(
                        output=output,
                        response_id=response_id,
                        model=request.model,
                        created=created_timestamp,
                        conversation=request.conversation,
                        include=request.include,
                        store=storing,
                        ttl=_response_ttl_seconds,
                        previous_response_id=request.previous_response_id,
                    ),
                    media_type="text/event-stream",
                )

            tool_calls_list = build_tool_calls(output)
            output_items = build_response_output_items(output, tool_calls_list)
            usage = build_response_usage(output) or ResponseUsage(
                input_tokens=0, output_tokens=0, total_tokens=0
            )

            response = Response(
                id=response_id,
                created_at=created_timestamp,
                expires_at=created_timestamp + _response_ttl_seconds
                if storing
                else None,
                model=request.model,
                status="completed",
                output=output_items,
                output_text=output.value or "",
                usage=usage,
                conversation=request.conversation,
                previous_response_id=request.previous_response_id,
            )

            # Store the response and full history for future previous_response_id use.
            if storing:
                _response_store[response_id] = _StoredResponse(
                    response=response,
                    history=_build_history_from_messages(messages, output.value or ""),
                )

            return response

        except ValueError as e:
            return create_openai_error_response(
                status_code=400,
                message=f"Invalid request: {e!s}",
                error_type="invalid_request_error",
            )
        except Exception:
            logger.exception("Unhandled error in responses endpoint")
            return create_openai_error_response(
                status_code=500,
                message="Internal server error",
                error_type="server_error",
            )

    endpoint.__name__ = f"responses_{module.__name__}_endpoint"
    return endpoint


def make_chat_endpoint(module):
    """Makes a chat endpoint using a custom module."""
    # Inspect serve function once at endpoint creation time
    serve_sig = inspect.signature(module.serve)
    accepts_format = "format" in serve_sig.parameters
    accepts_client_options = "client_options" in serve_sig.parameters
    is_async = inspect.iscoroutinefunction(module.serve)

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

            completion_id = "chatcmpl-" + "".join(
                secrets.choice(_BASE62) for _ in range(29)
            )
            created_timestamp = int(time.time())

            model_options = _build_model_options(request)

            # Handle response_format
            format_model: type[BaseModel] | None = None
            if request.response_format is not None:
                if request.response_format.type == "json_schema":
                    # json_schema presence is validated by ResponseFormat.model_validator
                    json_schema = cast(
                        JsonSchemaFormat, request.response_format.json_schema
                    )
                    try:
                        format_model = json_schema_to_pydantic(
                            json_schema.schema_, json_schema.name
                        )
                    except (ValueError, TypeError, RecursionError) as e:
                        message = (
                            "Invalid JSON schema: recursive $ref is not supported"
                            if isinstance(e, RecursionError)
                            else f"Invalid JSON schema: {e!s}"
                        )
                        return create_openai_error_response(
                            status_code=400,
                            message=message,
                            error_type="invalid_request_error",
                            param="response_format.json_schema.schema",
                        )
                # For "json_object" and "text", format_model remains None
                # Note: "json_object" mode is not yet implemented - the backend
                # receives no signal to produce JSON output (same as "text" mode)

            # Build kwargs for serve call
            serve_kwargs: dict[str, Any] = {
                "input": request.messages,
                "requirements": request.requirements,
                "model_options": model_options,
            }
            if accepts_format:
                serve_kwargs["format"] = format_model
            if accepts_client_options:
                serve_kwargs["client_options"] = _build_client_options(request)

            # Detect if serve is async or sync and handle accordingly
            if is_async:
                # It's async, await it directly
                output = await module.serve(**serve_kwargs)
            else:
                # It's sync, run in thread pool to avoid blocking event loop
                output = await asyncio.to_thread(module.serve, **serve_kwargs)

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

            tool_calls_list = build_tool_calls(output)
            tool_calls = (
                [
                    ChatCompletionMessageToolCall.model_validate(tc)
                    for tc in tool_calls_list
                ]
                if tool_calls_list
                else None
            )

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
                        finish_reason=extract_finish_reason(output),
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
        except Exception:
            logger.exception("Unhandled error in chat-completion handler")
            return create_openai_error_response(
                status_code=500,
                message="Internal server error",
                error_type="server_error",
            )

    endpoint.__name__ = f"chat_{module.__name__}_endpoint"
    return endpoint


@app.get("/v1/responses/{response_id}", response_model=Response | OpenAIErrorResponse)
async def get_response(response_id: str) -> Response | JSONResponse:
    """Retrieve a stored response by ID.

    Returns the completed ``Response`` object that was stored when
    ``store=True`` (the default) on the original ``POST /v1/responses``
    request.  Returns 404 if the response has expired or was created with
    ``store=False``.  The ``expires_at`` field on the response indicates
    the exact Unix timestamp when the entry will be evicted.
    """
    stored = _response_store.get(response_id)
    if stored is None:
        return create_openai_error_response(
            status_code=404,
            message=f"Response '{response_id}' not found or has expired.",
            error_type="invalid_request_error",
        )
    return stored.response


def run_server(
    script_path: str = "docs/examples/m_serve/example.py",
    host: str = "0.0.0.0",
    port: int = 8080,
    response_ttl: int = 1800,
):
    """Serve a FastAPI endpoint for a given script."""
    global _response_ttl_seconds
    _response_ttl_seconds = response_ttl

    module = load_module_from_path(script_path)

    app.add_api_route(
        "/v1/chat/completions",
        make_chat_endpoint(module),
        methods=["POST"],
        response_model=ChatCompletion | OpenAIErrorResponse,
    )
    app.add_api_route(
        "/v1/responses",
        make_responses_endpoint(module),
        methods=["POST"],
        response_model=Response | OpenAIErrorResponse,
    )

    ttl_minutes = response_ttl // 60
    typer.echo(
        f"Serving /v1/chat/completions and /v1/responses at http://{host}:{port}"
    )
    typer.echo(f"Response store TTL: {ttl_minutes} minutes")
    uvicorn.run(app, host=host, port=port)
