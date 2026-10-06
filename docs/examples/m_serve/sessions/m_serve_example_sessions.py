# pytest: ollama, e2e

"""Example server for multi-turn session demo via the Responses API.

The Responses API supports multi-turn conversations without the client
resending the full history. Each completed response is stored in the
server's in-memory session store. Pass the `id` from a previous
response as `previous_response_id` in the next request and the server
reconstructs the history automatically.

Usage:
    1. Start the server:
       m serve docs/examples/m_serve/sessions/m_serve_example_sessions.py

    2. Run the session client:
       python docs/examples/m_serve/sessions/client_responses_sessions.py

    Sessions expire after 30 minutes by default.  Use --response-ttl to
    change the TTL (in seconds):
       m serve docs/examples/m_serve/sessions/m_serve_example_sessions.py --response-ttl 3600
"""

from typing import Literal, cast

import mellea
from mellea.core import ModelOutputThunk
from mellea.serve import ChatMessage
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext

# Message accepts system/user/assistant/tool; "function" is a legacy role in
# ChatMessage that has no equivalent in Message.  Skip those entries.
_MESSAGE_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})
_MessageRole = Literal["system", "user", "assistant", "tool"]


def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict | None = None,
) -> ModelOutputThunk:
    """Pass the full conversation history to the model and return its reply."""
    # Build a ChatContext from the full message list (which includes prior turns
    # prepended by the server when previous_response_id is set).
    ctx = ChatContext()
    for msg in input:
        if msg.role not in _MESSAGE_ROLES:
            continue
        ctx = ctx.add(
            Message(
                role=cast(_MessageRole, msg.role), content=msg.get_text_content() or ""
            )
        )

    # The last message is the current user turn; the context carries the history.
    last_text = input[-1].get_text_content() or ""

    session = mellea.start_session(ctx=ctx)
    result = session.instruct(
        description=last_text, strategy=None, model_options=model_options
    )
    return result
