"""OpenAI API-compatible HTTP server that wraps mellea backends.

This module provides a FastAPI server implementing OpenAI's chat completions API,
allowing any OpenAI-compatible client to use mellea backends.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mellea.core import FancyLogger, ModelOutputThunk
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.session import MelleaSession, start_session


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None


class FunctionDefinition(BaseModel):
    """Definition of a function that can be called by the model."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """Request body for chat completions endpoint."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=1.0, ge=0, le=2)
    top_p: float | None = Field(default=1.0, ge=0, le=1)
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = Field(default=0, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=0, ge=-2, le=2)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    seed: int | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["none", "auto"] | dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class ChatCompletionMessage(BaseModel):
    """A message in a chat completion response."""

    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Choice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """Response from chat completions endpoint."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None


class ChatCompletionChunk(BaseModel):
    """A chunk in a streaming chat completion response."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict[str, Any]]


class Model(BaseModel):
    """Information about an available model."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = "list"
    data: list[Model]


class ServerConfig:
    """Configuration for the OpenAI-compatible server.

    Args:
        backend_name: Name of the mellea backend to use (e.g., "ollama", "openai").
        model_id: Model identifier to use with the backend.
        base_url: Base URL for the backend API (if applicable).
        api_key: API key for the backend (if applicable).
        default_model_options: Default model options to apply to all requests.
        backend_kwargs: Additional keyword arguments to pass to the backend.
    """

    def __init__(
        self,
        backend_name: str = "ollama",
        model_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model_options: dict[str, Any] | None = None,
        **backend_kwargs: Any,
    ):
        """Initialize server configuration."""
        self.backend_name = backend_name
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.default_model_options = default_model_options or {}
        self.backend_kwargs = backend_kwargs


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create a FastAPI application with OpenAI-compatible endpoints.

    Args:
        config: Server configuration. If None, uses default configuration.

    Returns:
        FastAPI application instance.
    """
    if config is None:
        config = ServerConfig()

    app = FastAPI(
        title="Mellea OpenAI-Compatible API",
        description="OpenAI API-compatible server powered by mellea",
        version="0.1.0",
    )

    # Initialize app state immediately (not in lifespan) for TestClient compatibility
    app.state.config = config
    app.state.sessions = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle."""
        yield
        # Cleanup sessions on shutdown
        for session in app.state.sessions.values():
            session.cleanup()

    # Set lifespan after state initialization
    app.router.lifespan_context = lifespan

    def get_or_create_session(model: str) -> MelleaSession:
        """Get or create a mellea session for the given model.

        Args:
            model: Model identifier from the request.

        Returns:
            MelleaSession instance.
        """
        config = app.state.config
        session_key = f"{config.backend_name}:{model}"

        if session_key not in app.state.sessions:
            # Build backend kwargs
            backend_kwargs = dict(config.backend_kwargs)
            if config.base_url:
                backend_kwargs["base_url"] = config.base_url
            if config.api_key:
                backend_kwargs["api_key"] = config.api_key

            # Start a new session
            # Use the client-requested model, falling back to config default only if not provided
            session = start_session(
                backend_name=config.backend_name,
                model_id=model or config.model_id,
                model_options=config.default_model_options,
                **backend_kwargs,
            )
            app.state.sessions[session_key] = session

        return app.state.sessions[session_key]

    def build_model_options(request: ChatCompletionRequest) -> dict[str, Any]:
        """Build model options from request parameters.

        Args:
            request: Chat completion request.

        Returns:
            Dictionary of model options.
        """
        model_opts: dict[str, Any] = {}

        if request.temperature is not None:
            model_opts["temperature"] = request.temperature
        if request.top_p is not None:
            model_opts["top_p"] = request.top_p
        if request.max_tokens is not None:
            model_opts["max_tokens"] = request.max_tokens
        if request.seed is not None:
            model_opts["seed"] = request.seed
        if request.stop is not None:
            model_opts["stop"] = request.stop
        if request.presence_penalty is not None:
            model_opts["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            model_opts["frequency_penalty"] = request.frequency_penalty
        if request.stream:
            model_opts["stream"] = True

        return model_opts

    async def stream_response(
        mot: ModelOutputThunk, completion_id: str, model: str, created: int
    ) -> AsyncIterator[str]:
        """Stream chat completion chunks.

        Args:
            mot: Model output thunk to stream from.
            completion_id: Unique completion ID.
            model: Model identifier.
            created: Creation timestamp.

        Yields:
            Server-sent event formatted strings.
        """
        try:
            prev_length = 0
            while not mot.is_computed():
                # Get the next chunk
                current_value = await mot.astream()

                # Extract only the new content since last iteration
                new_content = current_value[prev_length:] if current_value else ""
                prev_length = len(current_value) if current_value else 0

                if new_content:
                    chunk_data = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model,
                        choices=[
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": new_content},
                                "finish_reason": None,
                            }
                        ],
                    )
                    yield f"data: {chunk_data.model_dump_json()}\n\n"

            # Send final chunk with finish_reason
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model,
                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            FancyLogger.get_logger().error(f"Error during streaming: {e}")
            error_chunk = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest):
        """Handle chat completion requests.

        Args:
            request: Chat completion request body.

        Returns:
            Chat completion response or streaming response.

        Raises:
            HTTPException: If the request is invalid or processing fails.
        """
        try:
            # Get or create session
            session = get_or_create_session(request.model)

            # Build model options
            model_opts = build_model_options(request)

            # Extract the last user message content
            last_message = request.messages[-1]
            if not last_message.content:
                raise HTTPException(
                    status_code=400, detail="Last message must have content"
                )

            # Determine if we should use tool calls
            tool_calls = request.tools is not None and len(request.tools) > 0

            # Build conversation history - add all messages to session context
            for msg in request.messages[:-1]:
                if msg.content:
                    session.ctx = session.ctx.add(
                        Message(role=msg.role, content=msg.content)
                    )

            # Create the user message for the current request
            user_message = Message(role=last_message.role, content=last_message.content)

            # Generate completion using the backend
            # We use backend directly to get access to the ModelOutputThunk for streaming
            mot, new_ctx = await session.backend._generate_from_context(
                action=user_message,
                ctx=session.ctx,
                model_options=model_opts,
                tool_calls=tool_calls,
            )

            # Update session context with the new context
            session.ctx = new_ctx

            # Generate unique IDs
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created = int(time.time())

            # Handle streaming
            if request.stream:
                return StreamingResponse(
                    stream_response(mot, completion_id, request.model, created),
                    media_type="text/event-stream",
                )

            # Wait for completion
            await mot.avalue()

            # Extract usage information
            usage = None
            if mot.usage:
                usage = Usage(
                    prompt_tokens=mot.usage.get("prompt_tokens", 0),
                    completion_tokens=mot.usage.get("completion_tokens", 0),
                    total_tokens=mot.usage.get("total_tokens", 0),
                )

            # Build response
            response = ChatCompletion(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=mot.value,
                            tool_calls=(
                                [
                                    {
                                        "id": f"call_{uuid.uuid4().hex[:24]}",
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(call.args),
                                        },
                                    }
                                    for name, call in mot.tool_calls.items()
                                ]
                                if mot.tool_calls
                                else None
                            ),
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=usage,
            )

            return response

        except Exception as e:
            FancyLogger.get_logger().error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        """List available models.

        Returns:
            List of available models.
        """
        config = app.state.config
        # Return the configured model
        model_id = config.model_id or "default"
        return ModelList(
            data=[
                Model(
                    id=model_id, created=int(time.time()), owned_by=config.backend_name
                )
            ]
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint.

        Returns:
            Health status.
        """
        return {"status": "healthy"}

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config: ServerConfig | None = None,
    **uvicorn_kwargs: Any,
) -> None:
    """Run the OpenAI-compatible server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        config: Server configuration.
        uvicorn_kwargs: Additional keyword arguments for uvicorn.
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "uvicorn is required to run the server. "
            "Please install it with: pip install mellea[server]"
        ) from e

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)


if __name__ == "__main__":
    """Run the server with default configuration when executed directly."""
    print("Starting mellea OpenAI-compatible server with default configuration...")
    print("Backend: Ollama")
    print("Model: granite4:micro")
    print("")
    print("Server will be available at:")
    print("  - API: http://localhost:8000/v1/chat/completions")
    print("  - Docs: http://localhost:8000/docs")
    print("  - Health: http://localhost:8000/health")
    print("")
    print("Note: Requires Ollama to be running with granite4:micro model available")
    print("      Install with: ollama pull granite4:micro")
    print("")

    # Use default configuration with Ollama
    default_config = ServerConfig(backend_name="ollama", model_id="granite4:micro")

    run_server(host="0.0.0.0", port=8000, config=default_config)
