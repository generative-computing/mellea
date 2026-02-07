"""Chat primitives."""

from collections.abc import Mapping
from typing import Any, Literal

from ...core import (
    CBlock,
    Component,
    Context,
    ImageBlock,
    ModelOutputThunk,
    ModelToolCall,
    TemplateRepresentation,
)
from ...security import SecLevel
from .docs.document import Document


class Message(Component["Message"]):
    """A single Message in a Chat history.

    TODO: we may want to deprecate this Component entirely.
    The fact that some Component gets rendered as a chat message is `Formatter` miscellania.
    """

    Role = Literal["system", "user", "assistant", "tool"]

    def __init__(
        self,
        role: "Message.Role",
        content: str,
        *,
        images: None | list[ImageBlock] = None,
        documents: None | list[Document] = None,
    ):
        """Initializer for Chat messages.

        Args:
            role (str): The role that this message came from (e.g., user, assistant).
            content (str): The content of the message.
            images (list[ImageBlock]): The images associated with the message if any.
            documents (list[Document]): documents associated with the message if any.
        """
        self.role = role
        self.content = content  # TODO this should be private.
        self._content_cblock = CBlock(self.content)
        self._images = images
        self._docs = documents
        self._sec_level: SecLevel | None = None

    @property
    def images(self) -> None | list[str]:
        """Returns the images associated with this message as list of base 64 strings."""
        if self._images is not None:
            return [str(i) for i in self._images]
        return None

    @property
    def sec_level(self) -> SecLevel | None:
        """Get the security level for this Component.

        Returns:
            SecLevel if present, None otherwise
        """
        return self._sec_level

    def parts(self) -> list[Component | CBlock]:
        """Returns all of the constituent parts of an Instruction."""
        parts: list[Component | CBlock] = [self._content_cblock]
        if self._docs is not None:
            parts.extend(self._docs)
        if self._images is not None:
            parts.extend(self._images)
        return parts

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the content for a Language Model.

        Returns:
            The formatted output suitable for language models.
        """
        return TemplateRepresentation(
            obj=self,
            args={
                "role": self.role,
                "content": self._content_cblock,
                "images": self._images,
                "documents": self._docs,
            },
            template_order=["*", "Message"],
        )

    def __repr__(self) -> str:
        """Pretty representation of messages, because they are a special case."""
        images = []
        if self.images is not None:
            images = [f"{i[:20]}..." for i in self.images]

        docs = []
        if self._docs is not None:
            docs = [f"{doc.format_for_llm()[:10]}..." for doc in self._docs]
        return f'mellea.Message(role="{self.role}", content="{self.content}", images="{images}", documents="{docs}")'

    def _parse(self, computed: ModelOutputThunk) -> "Message":
        """Parse the model output into a Message."""
        # TODO: There's some specific logic for tool calls. Storing that here for now.
        # We may eventually need some generic parsing logic that gets run for all Component types...
        if computed.tool_calls is not None:
            # A tool was successfully requested.
            # Assistant responses for tool calling differ by backend. For the default formatter,
            # we put all of the function data into the content field in the same format we received it.

            # Chat backends should provide an openai-like object in the _meta chat response, which we can use to properly format this output.
            if "chat_response" in computed._meta:
                # Ollama.
                return Message(
                    role=computed._meta["chat_response"].message.role,
                    content=str(computed._meta["chat_response"].message.tool_calls),
                )
            elif "oai_chat_response" in computed._meta:
                # OpenAI and Watsonx.
                return Message(
                    role=computed._meta["oai_chat_response"]["choices"][0]["message"][
                        "role"
                    ],
                    content=str(
                        computed._meta["oai_chat_response"]["choices"][0][
                            "message"
                        ].get("tool_calls", [])
                    ),
                )
            else:
                # HuggingFace (or others). There are no guarantees on how the model represented the function calls.
                # Output it in the same format we received the tool call request.
                assert computed.value is not None
                return Message(role="assistant", content=computed.value)

        if "chat_response" in computed._meta:
            # Chat backends should provide an openai-like object in the _meta chat response, which we can use to properly format this output.
            return Message(
                role=computed._meta["chat_response"].message.role,
                content=computed._meta["chat_response"].message.content,
            )
        elif "oai_chat_response" in computed._meta:
            return Message(
                role=computed._meta["oai_chat_response"]["choices"][0]["message"][
                    "role"
                ],
                content=computed._meta["oai_chat_response"]["choices"][0]["message"][
                    "content"
                ],
            )
        else:
            assert computed.value is not None
            return Message(role="assistant", content=computed.value)


class ToolMessage(Message):
    """Adds the name field for function name."""

    def __init__(
        self,
        role: Message.Role,
        content: str,
        tool_output: Any,
        name: str,
        args: Mapping[str, Any],
        tool: ModelToolCall,
    ):
        """Initializer for Chat messages.

        Args:
            role: the role of this message. Most backends/models use something like tool.
            content: The content of the message; should be a stringified version of the tool_output.
            name: The name of the tool/function.
            args: The args required to call the function.
            tool_output: the output of the tool/function call.
            tool: the ModelToolCall representation.
        """
        super().__init__(role, content)
        self.name = name
        self.arguments = args
        self._tool_output = tool_output
        self._tool = tool

    def format_for_llm(self) -> TemplateRepresentation:
        """The same representation as Message with a name field added to args."""
        message_repr = super().format_for_llm()
        args = message_repr.args
        args["name"] = self.name

        return TemplateRepresentation(
            obj=self, args=args, template_order=["*", "Message"]
        )

    def __repr__(self) -> str:
        """Pretty representation of messages, because they are a special case."""
        return f'mellea.ToolMessage(role="{self.role}", content="{self.content}", name="{self.name}")'


def as_chat_history(ctx: Context) -> list[Message]:
    """Returns a list of Messages corresponding to a Context."""

    def _to_msg(c: CBlock | Component | ModelOutputThunk) -> Message | None:
        match c:
            case Message():
                return c
            case ModelOutputThunk():
                match c.parsed_repr:
                    case Message():
                        return c.parsed_repr
                    case _:
                        return None
            case _:
                return None

    all_ctx_events = ctx.as_list()
    if all_ctx_events is None:
        raise Exception("Trying to cast a non-linear history into a chat history.")
    else:
        history = [_to_msg(c) for c in all_ctx_events]
        assert None not in history, "Could not render this context as a chat history."
        return history  # type: ignore
