"""Interactive chat example with streaming responses.

This example shows how to build an interactive chat application where
the AI's responses are streamed incrementally for a better user experience.
"""

# pytest: ollama, llm

import asyncio

import mellea
from mellea.stdlib.context import ChatContext


async def interactive_chat():
    """Run an interactive chat session with streaming responses."""
    m = mellea.start_session(ctx=ChatContext())

    print("Chat with the AI (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        print("AI: ", end="", flush=True)

        # Stream the response
        thunk = await m.ainstruct(user_input, await_result=False)

        last_length = 0
        while not thunk.is_computed():
            current_value = await thunk.astream()
            # Print only the new portion
            new_content = current_value[last_length:]
            print(new_content, end="", flush=True)
            last_length = len(current_value)

        print()  # New line after response


if __name__ == "__main__":
    asyncio.run(interactive_chat())
