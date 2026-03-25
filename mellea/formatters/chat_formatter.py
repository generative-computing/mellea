"""``ChatFormatter`` for converting context histories to chat-message lists.

``ChatFormatter`` is the standard formatter used by mellea's legacy backends. Its
``to_chat_messages`` method linearises a sequence of ``Component`` and ``CBlock``
objects into ``Message`` objects with ``user``, ``assistant``, or ``tool`` roles,
handling ``ModelOutputThunk`` responses, image attachments, and parsed structured
outputs. Concrete backends call this formatter when preparing input for a chat
completion endpoint.
"""

from ..core import (
    CBlock,
    Component,
    Formatter,
    ModelOutputThunk,
    TemplateRepresentation,
)
from ..stdlib.components.chat import Message


class ChatFormatter(Formatter):
    """Formatter used by Legacy backends to format Contexts as Messages."""

    def to_chat_messages(self, cs: list[Component | CBlock]) -> list[Message]:
        """Convert a linearized chat history into a list of chat messages.

        Iterates over each element in the context history and converts it to a
        ``Message`` with an appropriate role. ``ModelOutputThunk`` instances are
        treated as assistant responses, while all other ``Component`` and
        ``CBlock`` objects default to the ``user`` role. Image attachments and
        parsed structured outputs are handled transparently.

        Args:
            cs (list[Component | CBlock]): The linearized sequence of context
                components and code blocks to convert.

        Returns:
            list[Message]: A list of ``Message`` objects ready for submission to
                a chat completion endpoint.
        """

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
