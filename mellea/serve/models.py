"""Types for Mellea's OpenAI-compatible server.

User-facing types (``ChatMessage``, ``MessageContent``, etc.) are used when
writing ``serve()`` functions.  Request/response types are used internally by
the FastAPI application and are also importable for testing.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel, model_validator

from mellea.core import ImageBlock, ImageUrlBlock
from mellea.helpers.openai_compatible_helpers import CompletionUsage

# ---------------------------------------------------------------------------
# User-facing message types (used when authoring a serve() function)
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    """Text content in a multimodal message."""

    type: Literal["text"]
    text: str


class ImageUrlContent(BaseModel):
    """Image URL content in a multimodal message.

    Supports both data URIs (base64-encoded images) and HTTP(S) URLs.
    """

    type: Literal["image_url"]
    image_url: dict[str, str]
    """Image URL object containing 'url' key and optional 'detail' key."""


# Union type for all content types
MessageContent = TextContent | ImageUrlContent


class ChatMessage(BaseModel):
    """Chat message with support for text-only or multimodal content.

    The content field can be:
    - A string (text-only, backward compatible)
    - None (for messages without content)
    - A list of content objects (multimodal: text, images)
    """

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | list[MessageContent] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None  # For function/tool messages

    def get_text_content(self) -> str:
        """Extract text content from message, handling both string and multimodal formats.

        Returns:
            Concatenated text from all TextContent items, or empty string if no text.
            Images are ignored (handled separately via extraction utilities).
        """
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            return " ".join(
                item.text for item in self.content if isinstance(item, TextContent)
            )
        return ""

    def get_image_urls(self) -> list[str]:
        """Extract image URLs from message content.

        Returns:
            List of image URL strings from all ImageUrlContent items.
            Empty list if content is a string or contains no images.
        """
        if not isinstance(self.content, list):
            return []
        urls = []
        for item in self.content:
            if isinstance(item, ImageUrlContent):
                url = item.image_url.get("url")
                if url:
                    urls.append(url)
        return urls

    def get_image_blocks(self) -> list[ImageBlock | ImageUrlBlock]:
        """Extract image blocks from message content.

        Returns:
            List of ``ImageBlock`` (for base64/data-URI images) or
            ``ImageUrlBlock`` (for http/https URLs) from all ImageUrlContent
            items. Empty list if content is a string or contains no images.

        Raises:
            ValueError: If an image URL is invalid or cannot be processed.
        """
        image_urls = self.get_image_urls()
        image_blocks: list[ImageBlock | ImageUrlBlock] = []
        for url in image_urls:
            if url.startswith(("http://", "https://")):
                image_blocks.append(ImageUrlBlock(url))
            elif url.startswith("data:"):
                try:
                    image_blocks.append(ImageBlock(url))
                except AssertionError as e:
                    # Raise ValueError for invalid data so the client gets a clear 400 error
                    # rather than silently processing a request without the expected images
                    raise ValueError(
                        f"Invalid image data: {url[:100]}{'...' if len(url) > 100 else ''}. "
                        f"Error: {e}"
                    ) from e
            else:
                raise ValueError(
                    f"Invalid image URL: {url[:100]}{'...' if len(url) > 100 else ''}. "
                    "Expected an http/https URL or a data: URI."
                )
        return image_blocks


# ---------------------------------------------------------------------------
# OpenAI-compatible request / response types (used by the FastAPI app)
# ---------------------------------------------------------------------------


class FunctionParameters(RootModel[dict[str, Any]]):
    """OpenAI-compatible function parameters as a bare JSON Schema object.

    Accepts a standard JSON Schema dict directly without wrapping.
    Example: {"type": "object", "properties": {...}, "required": [...]}
    """

    root: dict[str, Any]

    @model_validator(mode="after")
    def _reject_legacy_envelope(self) -> "FunctionParameters":
        """Reject legacy RootModel envelope pattern.

        Ensures parameters are sent as a bare JSON Schema object, not wrapped
        in a {"RootModel": {...}} envelope which would be invalid.
        """
        if set(self.root.keys()) == {"RootModel"}:
            raise ValueError(
                "Legacy {'RootModel': {...}} envelope is no longer accepted. "
                "Send parameters as a bare JSON Schema object."
            )
        return self


class FunctionDefinition(BaseModel):
    """OpenAI-compatible function definition."""

    name: str
    description: str | None = None
    parameters: FunctionParameters


class ToolFunction(BaseModel):
    """OpenAI-compatible tool (function) definition."""

    type: Literal["function"]
    function: FunctionDefinition


class JsonSchemaFormat(BaseModel):
    """JSON Schema definition for structured output."""

    name: str
    """Name of the schema."""

    schema_: dict[str, Any] = Field(alias="schema")
    """JSON Schema definition."""

    strict: bool | None = None
    """Accepted for OpenAI compatibility; currently ignored by `m serve`."""

    model_config = {"populate_by_name": True, "serialize_by_alias": True}


class ResponseFormat(BaseModel):
    """OpenAI-compatible response format specifier."""

    type: Literal["text", "json_object", "json_schema"]

    json_schema: JsonSchemaFormat | None = None
    """JSON Schema definition when type is 'json_schema'."""

    @model_validator(mode="after")
    def validate_json_schema_required(self) -> "ResponseFormat":
        """Validate that json_schema is provided when type is 'json_schema'."""
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("json_schema field is required when type is 'json_schema'")
        return self


class StreamOptions(BaseModel):
    """OpenAI-compatible streaming options.

    Controls behavior of streaming responses. Only applies when stream=True.
    """

    include_usage: bool = False
    """Whether to include usage statistics in the final streaming chunk.

    When True, the final chunk will include token usage information.
    When False (default), usage is excluded from streaming responses.
    For non-streaming requests, usage is always included regardless of this setting.
    """


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = {"extra": "allow"}

    model: str
    messages: list[ChatMessage]
    requirements: list[str | None] | None = Field(default_factory=list)
    functions: list[FunctionDefinition] | None = None
    function_call: Literal["none", "auto"] | dict[str, str] | None = None
    tools: list[ToolFunction] | None = None
    tool_choice: Literal["none", "auto"] | dict[str, Any] | None = None
    temperature: float | None = Field(default=1.0, ge=0, le=2)
    top_p: float | None = Field(default=1.0, ge=0, le=1)
    n: int | None = Field(default=1, ge=1)
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = Field(default=0, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=0, ge=-2, le=2)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    seed: int | None = None
    response_format: ResponseFormat | None = None
    stream_options: StreamOptions | None = None

    # For future/undocumented fields
    extra: dict[str, Any] = Field(default_factory=dict)


class ToolCallFunction(BaseModel):
    """Function details for a tool call."""

    name: str
    """The name of the function to call."""

    arguments: str
    """The arguments to call the function with, as a JSON string."""


class ChatCompletionMessageToolCall(BaseModel):
    """A tool call generated by the model (non-streaming)."""

    id: str
    """The ID of the tool call."""

    type: Literal["function"]
    """The type of the tool. Currently, only 'function' is supported."""

    function: ToolCallFunction
    """The function that the model called."""


class ToolCallFunctionDelta(BaseModel):
    """Function details for a streaming tool call delta.

    In streaming responses, function name and arguments may arrive across
    multiple chunks, so both fields are optional.
    """

    name: str | None = None
    """The name of the function to call (may be None in delta chunks)."""

    arguments: str | None = None
    """The arguments fragment for this delta (may be None in delta chunks)."""


class ChatCompletionMessageToolCallDelta(BaseModel):
    """A tool call delta in a streaming response.

    Per OpenAI streaming spec, each delta must include an index field that
    clients use to reassemble tool calls across chunks. The id, type, and
    function fields are optional since they may arrive incrementally.
    """

    index: int
    """The index of this tool call in the tool_calls array.

    Required for delta reassembly in OpenAI SDK and compatible clients.
    """

    id: str | None = None
    """The ID of the tool call (may be None in subsequent delta chunks)."""

    type: Literal["function"] | None = None
    """The type of the tool (may be None in subsequent delta chunks)."""

    function: ToolCallFunctionDelta | None = None
    """The function delta for this chunk (may be None in some chunks)."""


# Taking this from OpenAI types https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion.py,
class ChatCompletionMessage(BaseModel):
    """A chat completion message generated by the model."""

    content: str | None = None
    """The contents of the message."""

    refusal: str | None = None
    """The refusal message generated by the model."""

    role: Literal["assistant"]
    """The role of the author of this message."""

    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    """The tool calls generated by the model, such as function calls."""


class Choice(BaseModel):
    """A single completion choice."""

    index: int
    """The index of the choice in the list of choices."""

    message: ChatCompletionMessage
    """A chat completion message generated by the model."""

    finish_reason: (
        Literal["stop", "length", "content_filter", "tool_calls", "function_call"]
        | None
    ) = "stop"
    """The reason the model stopped generating tokens."""


class ChatCompletion(BaseModel):
    """An OpenAI-compatible chat completion response."""

    id: str
    """A unique identifier for the chat completion."""

    choices: list[Choice]
    """A list of chat completion choices.

    Can be more than one if `n` is greater than 1.
    """

    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created."""

    model: str
    """The model used for the chat completion."""

    object: Literal["chat.completion"]
    """The object type, which is always `chat.completion`."""

    system_fingerprint: str | None = None
    """This fingerprint represents the backend configuration that the model runs with."""

    usage: CompletionUsage | None = None
    """Usage statistics for the completion request."""


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    content: str | None = None
    """The content fragment in this chunk."""

    role: Literal["assistant"] | None = None
    """The role (only present in first chunk)."""

    refusal: str | None = None
    """The refusal message fragment, if any."""

    tool_calls: list[ChatCompletionMessageToolCallDelta] | None = None
    """The tool call deltas in this chunk.

    Each delta includes a required index field for reassembly by OpenAI SDK
    and compatible clients. The id, type, and function fields are optional
    since they may arrive incrementally across multiple chunks.
    """


class ChatCompletionChunkChoice(BaseModel):
    """A choice in a streaming chunk."""

    index: int
    """The index of the choice in the list of choices."""

    delta: ChatCompletionChunkDelta
    """The delta content for this chunk."""

    finish_reason: (
        Literal["stop", "length", "content_filter", "tool_calls", "function_call"]
        | None
    ) = None
    """The reason the model stopped generating tokens (only in final chunk)."""


class ChatCompletionChunk(BaseModel):
    """A chunk in a streaming chat completion response."""

    id: str
    """A unique identifier for the chat completion."""

    choices: list[ChatCompletionChunkChoice]
    """A list of chat completion choices."""

    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created."""

    model: str
    """The model used for the chat completion."""

    object: Literal["chat.completion.chunk"]
    """The object type, which is always `chat.completion.chunk`."""

    system_fingerprint: str | None = None
    """This fingerprint represents the backend configuration that the model runs with."""

    usage: CompletionUsage | None = None
    """Usage statistics for the final streaming chunk when available from the backend."""


class OpenAIError(BaseModel):
    """OpenAI API error object."""

    message: str
    """A human-readable error message."""

    type: str
    """The type of error (e.g., 'invalid_request_error', 'server_error')."""

    param: str | None = None
    """The parameter that caused the error, if applicable."""

    code: str | None = None
    """An error code, if applicable."""


class OpenAIErrorResponse(BaseModel):
    """OpenAI API error response wrapper."""

    error: OpenAIError
    """The error object."""
