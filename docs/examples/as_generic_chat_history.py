# pytest: unit
"""Convert a heterogeneous context to a generic chat history.

The as_generic_chat_history() function converts any Context into a list of
Messages, gracefully handling unknown component types by converting them to
strings. This is useful for working with mixed-type contexts or when you need
a more flexible interface than as_chat_history().
"""

from mellea.core import CBlock, ModelOutputThunk
from mellea.stdlib.components import Message, as_generic_chat_history
from mellea.stdlib.context import ChatContext


def basic_example() -> list[Message]:
    """Convert a standard Message-based context to chat history."""
    ctx = ChatContext()
    ctx = ctx.add(Message("user", "What is 2+2?"))
    ctx = ctx.add(Message("assistant", "2+2 equals 4."))

    history = as_generic_chat_history(ctx)
    assert len(history) == 2
    assert history[0].content == "What is 2+2?"
    assert history[1].content == "2+2 equals 4."
    return history


def with_heterogeneous_components() -> list[Message]:
    """Handle mixed component types gracefully.

    Unlike as_chat_history(), as_generic_chat_history() can handle any
    component type by converting unknown types to strings.
    """
    ctx = ChatContext()
    ctx = ctx.add(Message("user", "Summarize this"))
    ctx = ctx.add(CBlock("Some inline content to process"))
    mot = ModelOutputThunk(value="The summary is...")
    ctx = ctx.add(mot)

    history = as_generic_chat_history(ctx)
    assert len(history) == 3
    assert history[0].role == "user"
    assert history[1].role == "user"  # CBlock defaults to 'user'
    assert history[2].role == "assistant"  # MOT defaults to 'assistant'
    return history


def with_custom_formatter() -> list[Message]:
    """Use a custom formatter for ModelOutputThunk with unparsed content.

    You can provide a formatter function to customize how unparsed outputs
    or other unknown types are converted to strings.
    """

    def my_formatter(obj: object) -> str:
        return f"[Formatted: {type(obj).__name__}]"

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "Process this"))
    # Add a ModelOutputThunk with a non-Message parsed_repr
    mot = ModelOutputThunk(value="raw data")
    mot.parsed_repr = {"type": "dict", "data": "structured"}
    ctx = ctx.add(mot)

    history = as_generic_chat_history(ctx, formatter=my_formatter)
    assert len(history) == 2
    assert "[Formatted:" in history[1].content
    return history


if __name__ == "__main__":
    basic = basic_example()
    print("Basic example:")
    for msg in basic:
        print(f"  {msg.role}: {msg.content}")

    heterogeneous = with_heterogeneous_components()
    print("\nHeterogeneous example:")
    for msg in heterogeneous:
        print(f"  {msg.role}: {msg.content}")

    custom = with_custom_formatter()
    print("\nCustom formatter example:")
    for msg in custom:
        print(f"  {msg.role}: {msg.content}")
