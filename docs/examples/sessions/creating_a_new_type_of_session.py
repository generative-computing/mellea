from typing import Literal

from mellea import MelleaSession
from mellea.backends import Backend, BaseModelSubclass
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.base import CBlock, ChatContext, Context, ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import Requirement, reqify
from mellea.stdlib.safety.guardian import GuardianCheck, GuardianRisk

# This example shows how you might go about creating a new type of session.
# Here, we want to filter out potentially harmful chat messages from the user.


class ChatCheckingSession(MelleaSession):
    # Add new parameters to support applying requirements to chat messages and defining the filter strategy.
    def __init__(
        self,
        requirements: list[str | Requirement],
        backend: Backend,
        ctx: Context | None = None,
        check_immediately: bool = True,
    ):
        super().__init__(backend, ctx)
        self._requirements: list[Requirement] = [reqify(r) for r in requirements]

    # Override the chat method with our new filtering chat.
    def chat(
        self,
        content: str,
        role: Literal["system"]
        | Literal["user"]
        | Literal["assistant"]
        | Literal["tool"] = "user",
        *,
        user_variables: dict[str, str] | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> Message:
        # This simple check validates the user input against all the requirements. You could also
        # do something more complex here and explicitly indicate which checks failed.
        is_valid_input = self.validate(self._requirements, output=CBlock(content))
        if not all(is_valid_input):
            return Message(
                "assistant",
                "Incoming message did not pass chat requirements. Neither it nor this message will be appended to the context.",
            )

        # In this example, we still call the original MelleaSession.chat function. But there are some scenarios where you might not want to.
        # For example, if we were doing immediate filtering of the assistant's responses, we might want to wait to add the input and output message
        # to the context until the assistant message was checked. We could do that here by editing the context after the MelleaSession.chat was called,
        # or we could implement a completely new chat function where the context and generation is fully handled here.
        chat_msg = super().chat(
            content,
            role,
            user_variables=user_variables,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

        return chat_msg


m = ChatCheckingSession(
    requirements=[GuardianCheck("jailbreak"), GuardianCheck("profanity")],
    backend=OllamaModelBackend(),
    ctx=ChatContext(),
)

# You can run this code to see the immediate checks working.
while True:
    msg = input("User message: ")

    if msg == "":
        break

    result = m.chat(msg)
    print(result)
