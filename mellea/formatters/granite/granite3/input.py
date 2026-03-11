# SPDX-License-Identifier: Apache-2.0

"""Input processing code that is specific to the Granite 3 family of models, but not specific to a particular point release."""

# Standard
import datetime
from collections.abc import Callable

# First Party
from ..base.io import InputProcessor
from ..base.types import AssistantMessage, SystemMessage, ToolResultMessage, UserMessage
from ..granite3.types import Granite3ChatCompletion


class Granite3InputProcessor(InputProcessor):
    """Abstract base class for Granite 3.x input processors.

    Contains code that is common across point releases. See the classes for the individual point release for the APIs that perform input transformations.
    """

    @staticmethod
    def _make_system_message_start():
        """Return string that comes at the beginning of the system message.

        :returns: String that comes at the beginning of the system message that a
        Granite 3 model must receive at the beginning of the prompt for any completion
        request that does not provide a custom system message.

        Note that the original Jinja template tends to choose weird dates from the
        future for the "Today's date" part. Instead of replicating that behavior, we
        put today's actual date in that section of the prompt. This difference probably
        doesn't matter, since none of the supervised fine tuning data exercises
        knowledge cutoffs.
        """
        return f"""\
Knowledge Cutoff Date: April 2024.
Today's Date: {datetime.datetime.now().strftime("%B %d, %Y")}.
You are Granite, developed by IBM."""

    @staticmethod
    def _split_messages(
        chat_completion: Granite3ChatCompletion,
    ) -> tuple[SystemMessage | None, list]:
        """Separate the system message from other messages.

        :returns: Tuple of system message, if present, and remaining messages.
        """
        messages = chat_completion.messages

        # Validation code in the Inputs class should already have verified that there
        # are either zero or one system messages, and that the system message, if
        # present, occurs at position zero.
        if isinstance(messages[0], SystemMessage):
            # First message is a system message.
            return messages[0], messages[1:]
        return None, messages

    @staticmethod
    def _message_to_prompt_string(message: UserMessage | AssistantMessage) -> str:
        if isinstance(message, UserMessage):
            return (
                f"<|start_of_role|>user<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        if isinstance(message, AssistantMessage):
            # Note that we discard any tool calls in the message, per the Jinja
            # template.
            return (
                f"<|start_of_role|>assistant<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        if isinstance(message, ToolResultMessage):
            # Note that we discard the tool call ID, per the Jinja template.
            return (
                f"<|start_of_role|>tool<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        raise TypeError(f"Unexpected message type {type(message)}")

    @staticmethod
    def _build_controls_record(chat_completion: Granite3ChatCompletion) -> dict | None:
        """Build a Granite 3 controls record.

        Use the output control flags in ``inputs`` to build a version of the
        undocumented arbitrary JSON data regarding output controls that the Jinja
        template expected to see in the input for each chat completion request.

        :returns: A fake JSON record for "controls", or nothing if no output control
        flags were set.
        """
        kwargs = chat_completion._chat_template_kwargs()
        if not kwargs or not hasattr(kwargs, "controls") or not kwargs.controls:
            return None
        controls = kwargs.controls

        result = {}
        if controls.citations:
            # The following is a guess; we have no example data for this case.
            result["citations"] = True
        if controls.hallucinations:
            # The following is a guess; we have no example data for this case.
            result["hallucinations"] = True
        if controls.length is not None:
            result["length"] = controls.length
        if controls.originality is not None:
            result["originality"] = controls.originality

        if len(result) == 0:
            return None
        return result

    @classmethod
    def _sanitize(
        cls,
        chat_completion: Granite3ChatCompletion,
        remove_special_tokens: Callable[[str], str],
        parts: list[str] | str = "all",
    ) -> Granite3ChatCompletion:
        """Sanitize a chat completion request by removing special tokens.

        :chat_completion: Chat completion request with unsanitized inputs.
        :remove_special_tokens: Function that removes special tokens from the
            text string. Passed in subclass.
        :parts: The parts of the chat completion request to sanitize. Accepted
            values are "messages", "tools", "documents", and "all", which can be
            given individually or as part of a list. Defaults to "all".
        :returns: A new chat completion request with sanitized inputs.
        """
        # Make a copy of the chat completion object.
        chat_completion = Granite3ChatCompletion.model_validate(
            chat_completion.model_dump()
        )

        # Check given "parts" have expected values.
        sanitize_modes = ["messages", "tools", "documents", "all"]
        unsupported_parts = []
        if isinstance(parts, str):
            parts = [parts]
        for part in parts:
            if part not in sanitize_modes:
                unsupported_parts.append(part)
        if len(unsupported_parts) > 0:
            raise ValueError(
                "sanitize static method",
                "sanitize ({sanitize}) must be one of {sanitize_modes}",
                {"sanitize": unsupported_parts},
            )

        # Sanitize based on the given parts.
        if ("messages" in parts or "all" in parts) and chat_completion.messages:
            for message in chat_completion.messages:
                message.content = remove_special_tokens(message.content)
        if ("tools" in parts or "all" in parts) and chat_completion.tools:
            for tool in chat_completion.tools:
                tool.name = remove_special_tokens(tool.name)
                if tool.description:
                    tool.description = remove_special_tokens(tool.description)
                if tool.parameters:
                    new_params = {}
                    for k, v in tool.parameters.items():
                        kk = remove_special_tokens(k)
                        vv = remove_special_tokens(v)
                        if len(kk) > 0:
                            new_params[kk] = vv
                    tool.parameters = new_params
        docs = chat_completion._documents()
        if ("documents" in parts or "all" in parts) and docs:
            for document in docs:
                if document.doc_id and isinstance(document.doc_id, str):
                    document.doc_id = remove_special_tokens(document.doc_id)
                document.text = remove_special_tokens(document.text)

        return chat_completion
