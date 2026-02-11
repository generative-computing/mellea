"""Example of using Mellea Chat Model with LangChain chains."""

from langchain_core.prompts import ChatPromptTemplate
from mellea import start_session

from mellea_langchain import MelleaChatModel


def simple_chain_example():
    """Demonstrate simple chain usage."""
    print("\n" + "=" * 60)
    print("Simple Chain Example")
    print("=" * 60)

    # Create Mellea session and chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that explains technical concepts."),
            ("user", "{input}"),
        ]
    )

    # Create chain using LCEL (LangChain Expression Language)
    chain = prompt | chat_model

    # Run the chain
    print("\nQuestion: What is generative programming?")
    result = chain.invoke({"input": "What is generative programming?"})
    print(f"Response: {result.content[:200]}...")


def multi_step_chain_example():
    """Demonstrate multi-step chain."""
    print("\n" + "=" * 60)
    print("Multi-Step Chain Example")
    print("=" * 60)

    # Create Mellea session and chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # First chain: Generate a topic
    topic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a creative assistant."),
            ("user", "Suggest a {subject} topic in 3 words or less."),
        ]
    )
    topic_chain = topic_prompt | chat_model

    # Second chain: Explain the topic
    explain_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an educational assistant."),
            ("user", "Explain this topic in one sentence: {topic}"),
        ]
    )
    explain_chain = explain_prompt | chat_model

    # Run the chains
    print("\nStep 1: Generate a programming topic...")
    topic_result = topic_chain.invoke({"subject": "programming"})
    topic = topic_result.content
    print(f"Topic: {topic}")

    print("\nStep 2: Explain the topic...")
    explanation = explain_chain.invoke({"topic": topic})
    print(f"Explanation: {explanation.content}")


def main():
    """Run chain examples."""
    print("=" * 60)
    print("Mellea Chat Model - LangChain Chain Examples")
    print("=" * 60)

    # Run examples
    simple_chain_example()
    multi_step_chain_example()

    print("\n" + "=" * 60)
    print("Chain examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
