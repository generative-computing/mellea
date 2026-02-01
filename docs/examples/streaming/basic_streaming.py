"""Basic streaming example showing how to stream model outputs incrementally.

This example demonstrates the fundamental streaming capability in Mellea.
"""

# pytest: ollama, llm

import asyncio

import mellea


async def stream_story():
    """Stream a short story incrementally."""
    m = mellea.start_session()

    print("Streaming story generation...\n")

    # Get uncomputed thunk for streaming
    thunk = await m.ainstruct(
        "Write a short story about a robot learning to paint.", await_result=False
    )

    # Stream the output - astream() returns accumulated value so far
    last_length = 0
    while not thunk.is_computed():
        current_value = await thunk.astream()
        # Print only the new portion
        new_content = current_value[last_length:]
        print(new_content, end="", flush=True)
        last_length = len(current_value)

    print()  # New line at end


if __name__ == "__main__":
    asyncio.run(stream_story())
