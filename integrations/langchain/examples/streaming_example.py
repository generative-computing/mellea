"""Streaming example for Mellea Chat Model."""

import asyncio

from langchain_core.messages import HumanMessage
from mellea import start_session

from mellea_langchain import MelleaChatModel


async def async_streaming_example():
    """Demonstrate async streaming."""
    print("\n" + "=" * 60)
    print("Async Streaming Example")
    print("=" * 60)

    # Create Mellea session and chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Stream response
    print("\nStreaming response to: 'Write a short poem about AI'")
    print("-" * 60)

    messages = [HumanMessage(content="Write a short poem about AI")]

    async for chunk in chat_model.astream(messages):
        print(chunk.content, end="", flush=True)

    print("\n" + "-" * 60)


def sync_streaming_example():
    """Demonstrate sync streaming."""
    print("\n" + "=" * 60)
    print("Sync Streaming Example")
    print("=" * 60)

    # Create Mellea session and chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Stream response
    print("\nStreaming response to: 'Count from 1 to 5'")
    print("-" * 60)

    messages = [HumanMessage(content="Count from 1 to 5")]

    for chunk in chat_model.stream(messages):
        print(chunk.content, end="", flush=True)

    print("\n" + "-" * 60)


def main():
    """Run streaming examples."""
    print("=" * 60)
    print("Mellea Chat Model - Streaming Examples")
    print("=" * 60)

    # Run sync streaming
    sync_streaming_example()

    # Run async streaming
    asyncio.run(async_streaming_example())

    print("\n" + "=" * 60)
    print("Streaming examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
