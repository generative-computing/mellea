"""Chat primitives."""

from collections.abc import Mapping
from typing import Any, Literal

from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import (
    CBlock,
    Component,
    Context,
    Document,
    ImageBlock,
    ModelOutputThunk,
    ModelToolCall,
    TemplateRepresentation,
)


class Message(Component):
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
        # TODO this should replace _images.
        self._images_cblocks: list[CBlock] | None = None
        if self._images is not None:
            self._images_cblocks = [CBlock(str(i)) for i in self._images]
        self._docs = documents

    @property
    def images(self) -> None | list[str]:
        """Returns the images associated with this message as list of base 64 strings."""
        if self._images_cblocks is not None:
            return [str(i.value) for i in self._images_cblocks]
        return None

    def parts(self):
        """Returns all of the constituent parts of an Instruction."""
        FancyLogger.get_logger().error(
            "TODO: images are not handled correctly in the mellea core."
        )
        parts = [self._content_cblock]
        if self._docs is not None:
            parts.extend(self._docs)
        # TODO: we need to do this but images are not currently cblocks. This is captured in an issue on Jan 26 sprint. Leaving this code commented out for now.
        # if self._images is not None:
        #     parts.extend(self._images)
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
                "images": self._images_cblocks,
                "documents": self._docs,
            },
            template_order=["*", "Message"],
        )

    def __str__(self):
        """Pretty representation of messages, because they are a special case."""
        images = []
        if self.images is not None:
            images = [f"{i[:20]}..." for i in self.images]

        docs = []
        if self._docs is not None:
            docs = [f"{doc.format_for_llm()[:10]}..." for doc in self._docs]
        return f'mellea.Message(role="{self.role}", content="{self.content}", images="{images}", documents="{docs}")'


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

    def __str__(self):
        """Pretty representation of messages, because they are a special case."""
        return f'mellea.Message(role="{self.role}", content="{self.content}", name="{self.name}")'


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
