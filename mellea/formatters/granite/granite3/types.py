# SPDX-License-Identifier: Apache-2.0

"""Type definitions that are shared within the Granite 3 family of models."""

# Third Party
import pydantic
import pydantic_core

# First Party
from ..base.types import (
    AssistantMessage,
    ChatTemplateKwargs,
    Document,
    GraniteChatCompletion,
    NoDefaultsMixin,
    SystemMessage,
    UserMessage,
    VLLMExtraBody,
)


class Hallucination(pydantic.BaseModel):
    """Hallucination data as returned by the model output parser.

    Attributes:
        hallucination_id (str): Unique identifier for the hallucination entry.
        risk (str): Risk level of the hallucination, e.g. `"low"` or `"high"`.
        reasoning (str | None): Optional model-provided reasoning for why this
            sentence was flagged.
        response_text (str): The portion of the response text that is flagged.
        response_begin (int): Start character offset of `response_text` within
            the full response string.
        response_end (int): End character offset (exclusive) of `response_text`
            within the full response string.
    """

    hallucination_id: str
    risk: str
    reasoning: str | None = None
    response_text: str
    response_begin: int
    response_end: int


class Citation(pydantic.BaseModel):
    """Citation data as returned by the model output parser.

    Attributes:
        citation_id (str): Unique identifier assigned to this citation.
        doc_id (str): Identifier of the source document being cited.
        context_text (str): Verbatim text from the source document that is cited.
        context_begin (int): Start character offset of `context_text` within
            the source document.
        context_end (int): End character offset (exclusive) of `context_text`
            within the source document.
        response_text (str): The portion of the response text that makes this
            citation.
        response_begin (int): Start character offset of `response_text` within
            the response string.
        response_end (int): End character offset (exclusive) of `response_text`
            within the response string.
    """

    citation_id: str
    doc_id: str
    context_text: str
    context_begin: int
    context_end: int
    response_text: str
    response_begin: int
    response_end: int


class Granite3Controls(pydantic.BaseModel):
    """Control flags for Granite 3.x model output behaviour.

    Specifies which optional output features the model should produce, such as
    inline citations, hallucination risk annotations, response length constraints,
    and originality style.

    Attributes:
        citations (bool | None): When `True`, instructs the model to annotate
            factual claims with inline citation markers.
        hallucinations (bool | None): When `True`, instructs the model to
            append a list of sentences that may be hallucinated.
        length (str | None): Requested response length; must be `"short"`,
            `"long"`, or `None` for no constraint.
        originality (str | None): Requested response originality style; must be
            `"extractive"`, `"abstractive"`, or `None`.
    """

    citations: bool | None = None
    hallucinations: bool | None = None
    length: str | None = None  # Length output control variable
    originality: str | None = None

    @pydantic.field_validator("length", mode="after")
    @classmethod
    def _validate_length(cls, value: str | None) -> str | None:
        if value is None or value == "short" or value == "long":
            return value
        raise pydantic_core.PydanticCustomError(
            "length field validator",
            'length ({length}) must be "short" or "long" or None',
            {"length": value},
        )

    @pydantic.field_validator("originality", mode="after")
    @classmethod
    def _validate_originality(cls, value: str | None) -> str | None:
        if value is None or value == "extractive" or value == "abstractive":
            return value
        raise pydantic_core.PydanticCustomError(
            "originality field validator",
            'originality ({originality}) must be "extractive" or "abstractive" or None',
            {"originality": value},
        )


class Granite3Kwargs(ChatTemplateKwargs, NoDefaultsMixin):
    """Chat template keyword arguments specific to IBM Granite 3.x models.

    Extends :class:`ChatTemplateKwargs` with Granite 3-specific options for
    output control flags and chain-of-thought (thinking) mode.

    Attributes:
        controls (Granite3Controls | None): Optional output control flags that
            enable or configure citations, hallucination detection, response
            length, and originality style.
        thinking (bool): When `True`, enables chain-of-thought reasoning mode.
            Defaults to `False`.
    """

    controls: Granite3Controls | None = None
    thinking: bool = False


class Granite3ChatCompletion(GraniteChatCompletion):
    """Class that represents the inputs common to IBM Granite 3.x models.

    Class that represents the inputs that are common to models of the IBM Granite 3.x
    family.
    """

    def controls(self) -> Granite3Controls:
        """Return the Granite 3 controls record for this chat completion request.

        Returns:
            Granite3Controls: The controls record from the chat template kwargs,
                or an empty :class:`Granite3Controls` if none were specified.
        """
        kwargs = self.extra_body.chat_template_kwargs if self.extra_body else None
        if kwargs and isinstance(kwargs, Granite3Kwargs) and kwargs.controls:
            return kwargs.controls
        return Granite3Controls()

    def thinking(self) -> bool:
        """Return whether chain-of-thought thinking mode is enabled.

        Returns:
            bool: `True` if the `thinking` flag is set in the chat template
                kwargs; `False` otherwise.
        """
        kwargs = self.extra_body.chat_template_kwargs if self.extra_body else None
        return bool(kwargs and isinstance(kwargs, Granite3Kwargs) and kwargs.thinking)

    @pydantic.field_validator("extra_body")
    @classmethod
    def _validate_chat_template_kwargs(cls, extra_body: VLLMExtraBody) -> VLLMExtraBody:
        """Validate Granite 3 chat template kwargs and convert to dataclass.

        Validates kwargs that are specific to Granite 3 chat templates and converts
        the `chat_template_kwargs` field to a Granite 3-specific dataclass.

        Other arguments are currently passed through without checking.
        """
        if extra_body.chat_template_kwargs:
            kwargs_dict = extra_body.chat_template_kwargs.model_dump()
            extra_body.chat_template_kwargs = Granite3Kwargs.model_validate(kwargs_dict)
        return extra_body

    @pydantic.field_validator("messages")
    @classmethod
    def _validate_inputs_messages(cls, messages: list) -> list:
        # Make a copy so the validation code below can mutate the messages list but pass
        # through the original value. The caller also might have a pointer to the list.
        original_messages = messages
        messages = messages.copy()

        # There is no supervised fine tuning data for the case of zero messages.
        # Models are not guaranteed to produce a valid response if there are zero
        # messages.
        if len(messages) == 0:
            raise ValueError(
                "No messages. Model behavior for this case is not defined."
            )

        # The first message, and only the first message, may be the system message.
        first_message_is_system_message = isinstance(messages[0], SystemMessage)
        if first_message_is_system_message:
            messages = messages[1:]
            # If there is a system message, there must be at least one more user or
            # assistant message.
            if len(messages) == 0:
                raise ValueError(
                    "Input contains only a system message. Model behavior for this "
                    "case is not defined."
                )

        # The first message that is not a system message must be
        # either a user or assistant message.
        if not isinstance(messages[0], UserMessage | AssistantMessage):
            if first_message_is_system_message:
                raise ValueError(
                    f"First message after system message must be a user or "
                    f"assistant message. Found type {type(messages[0])}"
                )
            raise ValueError(
                f"First message must be a system, user, or assistant "
                f"Found type {type(messages[0])}"
            )

        # Undocumented constraint: All other messages form a conversation that
        # alternates strictly between user and assistant, possibly with tool calls
        # after an assistant turn and before the next user turn.
        # TODO: Validate this invariant.

        # Pydantic will use the value that this validator returns as the value of the
        # messages field. Undo any changes that we made during validation and return
        # the original value.
        return original_messages


class Granite3AssistantMessage(AssistantMessage):
    """An assistant message with Granite 3 specific fields.

    Attributes:
        reasoning_content (str | None): Optional chain-of-thought reasoning text
            produced before the final response.
        citations (list[Citation] | None): Optional list of citations parsed from
            the model output.
        documents (list[Document] | None): Optional list of documents referenced
            in the model output.
        hallucinations (list[Hallucination] | None): Optional list of hallucination
            annotations parsed from the model output.
        stop_reason (str | None): Optional reason the model stopped generating.
    """

    reasoning_content: str | None = None
    citations: list[Citation] | None = None
    documents: list[Document] | None = None
    hallucinations: list[Hallucination] | None = None
    stop_reason: str | None = None

    raw_content: str | None = pydantic.Field(
        default=None,
        description=(
            "Raw response content without any parsing, for debugging and "
            "re-serialization."
        ),
    )
