"""Advanced streaming example with error handling and buffering.

This example demonstrates production-ready streaming patterns including:
- Error handling for network issues
- Buffering for smoother UI updates
- Timeout handling
- Graceful degradation
"""

# pytest: ollama, llm

import asyncio

import mellea


async def stream_with_error_handling():
    """Stream with comprehensive error handling."""
    m = mellea.start_session()

    print("Streaming with error handling...\n")

    try:
        thunk = await m.ainstruct(
            "Write a technical explanation of how neural networks learn.",
            await_result=False,
        )

        last_length = 0
        while not thunk.is_computed():
            current_value = await thunk.astream()
            new_content = current_value[last_length:]
            print(new_content, end="", flush=True)
            last_length = len(current_value)
        print()

    except asyncio.TimeoutError:
        print("\n[Error: Request timed out]")
    except Exception as e:
        print(f"\n[Error: {e}]")


async def stream_with_buffering():
    """Stream with buffering for smoother output."""
    m = mellea.start_session()

    print("\nStreaming with buffering...\n")

    thunk = await m.ainstruct(
        "Explain quantum computing in simple terms.", await_result=False
    )

    last_length = 0
    buffer = []
    buffer_size = 50  # Buffer 50 characters before displaying

    while not thunk.is_computed():
        current_value = await thunk.astream()
        new_content = current_value[last_length:]
        buffer.append(new_content)
        last_length = len(current_value)

        # Flush buffer when it reaches the size threshold
        if sum(len(s) for s in buffer) >= buffer_size:
            print("".join(buffer), end="", flush=True)
            buffer = []

    # Flush remaining buffer
    if buffer:
        print("".join(buffer), end="", flush=True)
    print()


async def compare_streaming_vs_blocking():
    """Compare streaming vs blocking behavior."""
    m = mellea.start_session()

    print("\n" + "=" * 60)
    print("COMPARISON: Streaming vs Blocking")
    print("=" * 60)

    # Blocking (default behavior)
    print("\n1. Blocking mode (await_result=True, default):")
    print("   Waiting for complete response...")
    result = await m.ainstruct(
        "Write a haiku about programming.",
        await_result=True,  # This is the default
    )
    print(f"   Result: {result.value}")

    # Streaming
    print("\n2. Streaming mode (await_result=False):")
    print("   Tokens appear as generated: ", end="", flush=True)
    thunk = await m.ainstruct("Write a haiku about programming.", await_result=False)

    last_length = 0
    while not thunk.is_computed():
        current_value = await thunk.astream()
        new_content = current_value[last_length:]
        print(new_content, end="", flush=True)
        last_length = len(current_value)
    print()


async def main():
    """Run all advanced streaming examples."""
    await stream_with_error_handling()
    await stream_with_buffering()
    await compare_streaming_vs_blocking()


if __name__ == "__main__":
    asyncio.run(main())

# Made with Bob
