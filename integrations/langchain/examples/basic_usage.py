"""Basic usage example of Mellea Chat Model for LangChain."""

from langchain_core.messages import HumanMessage, SystemMessage
from mellea import start_session

from mellea_langchain import MelleaChatModel


def main():
    """Demonstrate basic usage of MelleaChatModel."""
    print("=" * 60)
    print("Mellea Chat Model for LangChain - Basic Usage Example")
    print("=" * 60)

    # Create Mellea session (uses Ollama by default)
    print("\n1. Creating Mellea session...")
    m = start_session()
    print("   ✓ Mellea session created")

    # Create LangChain chat model
    print("\n2. Creating LangChain chat model...")
    chat_model = MelleaChatModel(mellea_session=m, model_name="mellea-ollama")
    print("   ✓ Chat model created")

    # Simple chat
    print("\n3. Simple chat completion...")
    messages = [HumanMessage(content="What is generative programming?")]
    response = chat_model.invoke(messages)
    print(f"   Question: {messages[0].content}")
    print(f"   Response: {response.content[:200]}...")

    # Chat with system message
    print("\n4. Chat with system message...")
    messages = [
        SystemMessage(content="You are a helpful assistant that explains concepts concisely."),
        HumanMessage(content="Explain what Mellea is in one sentence."),
    ]
    response = chat_model.invoke(messages)
    print(f"   Question: {messages[1].content}")
    print(f"   Response: {response.content}")

    # Multi-turn conversation
    print("\n5. Multi-turn conversation...")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi, my name is Alice."),
    ]
    response1 = chat_model.invoke(messages)
    print(f"   User: {messages[1].content}")
    print(f"   Assistant: {response1.content}")

    messages.append(response1)
    messages.append(HumanMessage(content="What's my name?"))
    response2 = chat_model.invoke(messages)
    print(f"   User: {messages[3].content}")
    print(f"   Assistant: {response2.content}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
