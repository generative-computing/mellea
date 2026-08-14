# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`ChatFormatter` for converting context histories to chat-message lists.

`ChatFormatter` is the standard formatter used by mellea's legacy backends. Its
`to_chat_messages` method linearises a sequence of `Component` and `CBlock`
objects into `Message` objects with `user`, `assistant`, or `tool` roles,
handling `ModelOutputThunk` responses, image attachments, and parsed structured
outputs. Concrete backends call this formatter when preparing input for a chat
completion endpoint.
"""

from ..core import Component, Formatter, ModelOutputThunk, Span, TemplateRepresentation
from ..stdlib.components.chat import Message, message_from_template_representation


class ChatFormatter(Formatter):
    """Formatter used by Legacy backends to format Contexts as Messages."""

    def to_chat_messages(self, cs: list[Span]) -> list[Message]:
        """Convert a linearized chat history into a list of chat messages.

        Iterates over each element in the context history and converts it to a
        `Message` with an appropriate role. `ModelOutputThunk` instances are
        treated as assistant responses, while all other `Component` and
        `CBlock` objects default to the `user` role. A `Component` may override
        this positional guess by setting `role` on the `TemplateRepresentation`
        returned from its `format_for_llm`, and a component with `role="tool"`
        may additionally declare `tool_name`/`tool_args`/`tool_call_id` to be
        rendered as a `ToolMessage`. Image attachments and parsed structured
        outputs are handled transparently.

        Args:
            cs (list[Span]): The linearized sequence of context
                components, content blocks, and model outputs to convert.

        Returns:
            list[Message]: A list of `Message` objects ready for submission to
                a chat completion endpoint.

        Raises:
            ValueError: If a component declares a `role` (via its
                `TemplateRepresentation`) outside `Message.Role`; role
                validation is deferred to `Message`/`ToolMessage`.
        """

        def _to_msg(c: Span) -> Message:
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
                    # A Message (or ToolMessage) is already a fully-formed chat
                    # message; return it verbatim so subtype-specific state such as
                    # `ToolMessage._tool.tool_call_id` survives to the backend payload.
                    return c
                case Component():
                    tr = c.format_for_llm()
                    if isinstance(tr, TemplateRepresentation):
                        # A component may declare its own role and tool metadata via
                        # its template representation; honor them over the positional
                        # guess. Role validation is deferred to `Message`/`ToolMessage`,
                        # which raise ValueError for anything outside `Message.Role`.
                        return message_from_template_representation(
                            tr, default_role=role, content=self.print(c)
                        )
                    return Message(role=role, content=self.print(c))
                case _:
                    return Message(role=role, content=self.print(c))

        return [_to_msg(c) for c in cs]
