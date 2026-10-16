# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Re-exported so callers can import from cli.serve.models instead of mellea.serve.models
__all__ = [
    "ChatMessage",
    "ContextManagementConfig",
    "ImageUrlContent",
    "InputAudioContent",
    "InputAudioData",
    "InputContent",
    "MessageContent",
    "OutputTextContent",
    "PromptCacheOptions",
    "Response",
    "ResponseError",
    "ResponseFunctionCall",
    "ResponseInputItem",
    "ResponseOutputItem",
    "ResponseOutputMessage",
    "ResponseRequest",
    "ResponseTextConfig",
    "ResponseTool",
    "ResponseUsage",
    "TextContent",
]

from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator

from mellea.helpers.openai_compatible_helpers import (
    CompletionUsage,
    OutputTextContent,
    ResponseFunctionCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseUsage,
)
from mellea.serve.models import (
    ChatMessage,
    ImageUrlContent,
    InputAudioContent,
    InputAudioData,
    MessageContent,
    TextContent,
)


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
    name: str
    description: str | None = None
    parameters: FunctionParameters
    strict: bool | None = None


class ToolFunction(BaseModel):
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
    """Response format configuration for structured outputs.

    Accepts both the Chat Completions nested shape
    (`{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}`)
    and the flat Responses API / SDK shape
    (`{"type": "json_schema", "name", "schema", "strict"}`). Flat fields are
    folded into `json_schema` at parse time so consumers only ever see the
    nested shape.
    """

    type: Literal["text", "json_object", "json_schema"]

    json_schema: JsonSchemaFormat | None = None
    """JSON Schema definition when type is 'json_schema'."""

    # Flat (Responses API / SDK) json_schema fields. Accepted at the edge
    # and folded into `json_schema` by _fold_flat_json_schema; excluded from
    # serialization so consumers only ever see the nested shape.
    name: str | None = Field(default=None, exclude=True)
    schema_: dict[str, Any] | None = Field(default=None, alias="schema", exclude=True)
    strict: bool | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _fold_flat_json_schema(self) -> "ResponseFormat":
        if (
            self.type == "json_schema"
            and self.json_schema is None
            and self.name is not None
        ):
            # Construct via the "schema" alias (mypy's pydantic plugin
            # expects the alias for aliased fields).
            self.json_schema = JsonSchemaFormat(
                name=self.name, schema=self.schema_ or {}, strict=self.strict
            )
        return self

    @model_validator(mode="after")
    def validate_json_schema_required(self) -> "ResponseFormat":
        """Validate that json_schema is provided when type is 'json_schema'."""
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError(
                "json_schema field is required when type is 'json_schema' "
                "(send the nested 'json_schema' object or flat 'name'/'schema' keys)"
            )
        return self


class ResponseTextConfig(BaseModel):
    """Configuration for the Responses API `text` request parameter.

    Mirrors the OpenAI SDK's `text` argument
    (`text={"format": {...}, "verbosity": "low"}`). `format` is lifted into
    the top-level `format` field of `ResponseRequest` at parse time.
    """

    format: ResponseFormat | None = None

    verbosity: Literal["low", "medium", "high"] | None = None
    """Accepted for OpenAI compatibility; currently ignored by `m serve`."""


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
    content: str | None = None
    """The contents of the message."""

    refusal: str | None = None
    """The refusal message generated by the model."""

    role: Literal["assistant"]
    """The role of the author of this message."""

    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    """The tool calls generated by the model, such as function calls."""


class Choice(BaseModel):
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


# ---------------------------------------------------------------------------
# Responses API models
# ---------------------------------------------------------------------------


class InputContent(BaseModel):
    """Content part in a Responses API input item."""

    type: Literal["input_text", "input_image", "input_file"]
    text: str | None = None
    image_url: str | None = None
    file_id: str | None = None


class ResponseInputItem(BaseModel):
    """A single input item for the Responses API.

    Accepts message items (`{"role": ..., "content": ...}`, optionally with
    an explicit `type: "message"`) and tool-result items
    (`{"type": "function_call_output", "call_id": ..., "output": ...}`) as
    sent by the OpenAI SDK.
    """

    type: Literal["message", "function_call_output"] | None = None
    role: Literal["user", "assistant", "developer", "system", "tool"] | None = None
    content: str | list[InputContent] | None = None
    tool_call_id: str | None = None
    tool_calls: list[Any] | None = None
    # function_call_output shape:
    call_id: str | None = None
    output: str | list[InputContent] | None = None

    @model_validator(mode="after")
    def _require_role_or_function_call_output(self) -> "ResponseInputItem":
        if self.role is None and self.type != "function_call_output":
            raise ValueError(
                "Input item requires a 'role' (message item) or "
                "type='function_call_output' (tool result)."
            )
        return self


class ResponseTool(BaseModel):
    """Tool declaration accepted by the Responses API.

    Accepts both the OpenAI Responses API / SDK flat function shape
    (`{"type": "function", "name", "parameters", "strict"}`) and the nested
    Chat Completions shape (`{"type": "function", "function": {...}}`).
    Flat fields are folded into `function` at parse time so `serve()` always
    receives the nested shape.

    Only `type: "function"` is currently supported by m serve. The other
    types (`web_search`, `file_search`, `mcp`, `code_interpreter`)
    are accepted by the schema so that requests are parsed and rejected with
    a clear 400 error rather than a Pydantic validation failure.
    """

    type: Literal["function", "web_search", "file_search", "mcp", "code_interpreter"]
    function: FunctionDefinition | None = None
    # Flat (Responses API / SDK) function-tool fields. Accepted at the edge
    # and folded into `function` by _fold_flat_function; excluded from
    # serialization so `serve()` only ever sees the nested shape.
    name: str | None = Field(default=None, exclude=True)
    description: str | None = Field(default=None, exclude=True)
    parameters: dict[str, Any] | None = Field(default=None, exclude=True)
    strict: bool | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _fold_flat_function(self) -> "ResponseTool":
        if self.type == "function" and self.function is None and self.name is not None:
            self.function = FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=FunctionParameters(root=self.parameters or {}),
                strict=self.strict,
            )
        return self


class ContextManagementConfig(BaseModel):
    """Context compaction configuration."""

    compact_threshold: float | None = None
    strategy: Literal["auto", "manual"] | None = None


class PromptCacheOptions(BaseModel):
    """Prompt caching settings."""

    enabled: bool | None = None
    breakpoint: str | None = None


class ResponseRequest(BaseModel):
    """Request body for POST /v1/responses."""

    model_config = {"extra": "allow"}

    model: str
    """Model to use (e.g. gpt-5.6, o1)."""

    input: str | list[ResponseInputItem]
    """Text, image, or file inputs."""

    instructions: str | list[ResponseInputItem] | None = None
    """System/developer context for the response."""

    requirements: list[str | None] | None = Field(default_factory=list)
    """Mellea requirements - passed separately to serve()."""

    tools: list[ResponseTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None
    max_output_tokens: int | None = None
    temperature: float | None = Field(default=1.0, ge=0, le=2)
    top_p: float | None = Field(default=1.0, ge=0, le=1)
    stream: bool | None = False
    store: bool | None = True
    conversation: str | None = None
    """Conversation id. The SDK's object form ({"id": ...}) is accepted and
    collapsed to its id string at parse time."""

    previous_response_id: str | None = None
    background: bool | None = False
    include: list[str] | None = None
    parallel_tool_calls: bool | None = True
    context_management: (
        list[ContextManagementConfig] | ContextManagementConfig | None
    ) = None
    """Accepted for OpenAI compatibility (the SDK sends a list); ignored."""
    prompt_cache_options: PromptCacheOptions | None = None
    format: ResponseFormat | None = None
    """Response format configuration for structured outputs.

    The OpenAI SDK sends structured output as `text={"format": {...}}`; the
    flat `text.format` is lifted into this field at parse time. This
    top-level field is the canonical (and alias) location, accepting both
    the nested and flat shapes.

    Configuring `{ "type": "json_schema" }` enables Structured Outputs,
    which ensures the model will match your supplied JSON schema.
    The default format is `{ "type": "text" }`.
    """

    # SDK `text` parameter. Accepted at the edge and lifted into `format`
    # by _lift_text_format; excluded from serialization so serve() only ever
    # sees the canonical `format` field.
    text: ResponseTextConfig | None = Field(default=None, exclude=True)

    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation", mode="before")
    @classmethod
    def _normalize_conversation(cls, value: Any) -> Any:
        # The OpenAI SDK accepts conversation={"id": ...} alongside a bare id
        # string; collapse the object form to its id so the field stays a
        # string and the conversation echo round-trips.
        if isinstance(value, dict) and "id" in value:
            return value["id"]
        return value

    @model_validator(mode="after")
    def _normalize_tool_choice(self) -> "ResponseRequest":
        # The OpenAI Responses API / SDK sends a specific-function choice as
        # the flat {"type": "function", "name": ...}; the Chat Completions
        # shape is {"type": "function", "function": {"name": ...}}. Normalize
        # the flat form to the nested one so serve() (and the shipped
        # examples) read a single consistent shape. String choices
        # ("none"/"auto"/"required") pass through unchanged.
        tc = self.tool_choice
        if (
            isinstance(tc, dict)
            and tc.get("type") == "function"
            and "function" not in tc
            and tc.get("name") is not None
        ):
            self.tool_choice = {"type": "function", "function": {"name": tc["name"]}}
        return self

    @model_validator(mode="after")
    def _lift_text_format(self) -> "ResponseRequest":
        # The OpenAI Responses API / SDK sends structured output as
        # text={"format": {...}} rather than a top-level `format`. Lift the
        # nested format into the canonical top-level field so the endpoint's
        # format handling sees one consistent location. When both are sent,
        # the top-level `format` wins (only lift when it is None), matching
        # the nested-wins rule of ResponseTool._fold_flat_function.
        if (
            self.format is None
            and self.text is not None
            and self.text.format is not None
        ):
            self.format = self.text.format
        return self


class ResponseError(BaseModel):
    """Error object embedded in a failed Responses API response."""

    code: str
    message: str
    param: str | None = None


class Response(BaseModel):
    """A completed response from POST /v1/responses."""

    id: str
    """Unique response identifier (resp_…)."""

    created_at: int
    """Unix timestamp of when the response was created."""

    expires_at: int | None = None
    """Unix timestamp after which this response is no longer retrievable.

    Set to `created_at + TTL` when `store=True` (the default); `None`
    when `store=False` because the response is never persisted and cannot
    be retrieved via `GET /v1/responses/{id}`. The TTL is controlled by
    the `--response-ttl` server flag (default: 1800 seconds / 30 minutes).
    """

    model: str
    """Model used to generate the response."""

    status: Literal["completed", "failed", "in_progress", "incomplete"]
    output: list[ResponseOutputMessage | ResponseFunctionCall | ResponseOutputItem]
    """Typed output items (messages, function calls, etc.).

    Uses a union type to ensure Pydantic serializes subclass-specific fields
    (e.g., `content` and `role` for messages, `name` and `arguments`
    for function calls) rather than stripping them to the base class schema.
    """

    output_text: str
    """Convenience field: concatenated text from all output message items."""

    error: ResponseError | None = None
    incomplete_details: dict[str, Any] | None = None
    usage: ResponseUsage
    conversation: str | None = None
    previous_response_id: str | None = None
