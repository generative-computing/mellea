"""ChatFormatter."""

from ..core import (
    CBlock,
    Component,
    Formatter,
    ModelOutputThunk,
    TemplateRepresentation,
)
from ..stdlib.components import Message


class ChatFormatter(Formatter):
    """Formatter used by Legacy backends to format Contexts as Messages."""

    def to_chat_messages(self, cs: list[Component | CBlock]) -> list[Message]:
        """Helper method that converts a linearized chat history into a list of messages. The purpose of this helper is to prepare a sequence of Messages for input to a chat endpoint."""

        def _to_msg(c: Component | CBlock) -> Message:
            role: Message.Role = "user"  # default to `user`; see ModelOutputThunk below for when the role changes.

            # Check if it's a ModelOutputThunk first since that changes what we should be printing
            # as the message content.
            if isinstance(c, ModelOutputThunk):
                role = "assistant"  # ModelOutputThunks should always be responses from a model.

                assert c.is_computed()
                assert (
                    c.value is not None
                )  # This is already entailed by c.is_computed(); the line is included here to satisfy the type-checker.

                if c.parsed_repr is not None:
                    if isinstance(c.parsed_repr, Component):
                        # Only use the parsed_repr if it's something that we know how to print.
                        c = c.parsed_repr  # This might be a message.
                    else:
                        # Otherwise, explicitly stringify it.
                        c = Message(role=role, content=str(c.parsed_repr))
                else:
                    c = Message(role=role, content=c.value)  # type: ignore

            match c:
                case Message():
                    return c
                case Component():
                    images = None
                    tr = c.format_for_llm()
                    if isinstance(tr, TemplateRepresentation):
                        images = tr.images

                    # components can have images
                    return Message(role=role, content=self.print(c), images=images)
                case _:
                    return Message(role=role, content=self.print(c))

        return [_to_msg(c) for c in cs]
