"""Example demonstrating Mellea's requirements and strategy capabilities in LangChain.

This example shows how to use:
- Requirements: Define post-conditions for LLM outputs
- Strategy: Use RejectionSamplingStrategy to validate and retry
- Sampling results: Access detailed validation information
"""

from langchain_core.messages import HumanMessage
from mellea import start_session
from mellea.stdlib.requirements import check, req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea_langchain import MelleaChatModel


def example_basic_requirements():
    """Example 1: Basic requirements without validation strategy."""
    print("=" * 60)
    print("Example 1: Basic Requirements (no validation)")
    print("=" * 60)

    # Create Mellea session and LangChain chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Use requirements in model_options
    response = chat_model.invoke(
        [HumanMessage(content="Write a short email to Olivia thanking her for organizing events.")],
        model_options={
            "requirements": [
                "The email should have a salutation",
                "Use only lower-case letters",
            ]
        },
    )

    print(response.content)
    print()


def example_requirements_with_strategy():
    """Example 2: Requirements with RejectionSamplingStrategy."""
    print("=" * 60)
    print("Example 2: Requirements with Rejection Sampling")
    print("=" * 60)

    # Create Mellea session and LangChain chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Use requirements with a sampling strategy
    response = chat_model.invoke(
        [HumanMessage(content="Write a short email to Olivia thanking her for organizing events.")],
        model_options={
            "requirements": [
                "The email should have a salutation",
                "Use only lower-case letters",
            ],
            "strategy": RejectionSamplingStrategy(loop_budget=5),
            "return_sampling_results": True,
        },
    )

    print(response.content)
    print()


def example_custom_validation():
    """Example 3: Custom validation functions."""
    print("=" * 60)
    print("Example 3: Custom Validation Functions")
    print("=" * 60)

    # Create Mellea session and LangChain chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Define requirements with custom validation
    requirements = [
        req("The email should have a salutation"),
        req(
            "Use only lower-case letters",
            validation_fn=simple_validate(lambda x: x.lower() == x),
        ),
        check("Do not mention purple elephants."),
    ]

    response = chat_model.invoke(
        [HumanMessage(content="Write a short email to Olivia thanking her for organizing events.")],
        model_options={
            "requirements": requirements,
            "strategy": RejectionSamplingStrategy(loop_budget=5),
            "return_sampling_results": True,
        },
    )

    print(response.content)
    print()


def example_with_langchain_chain():
    """Example 4: Using requirements in a LangChain chain."""
    print("=" * 60)
    print("Example 4: Requirements in LangChain Chain")
    print("=" * 60)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    # Create Mellea session and LangChain chat model
    m = start_session()
    chat_model = MelleaChatModel(mellea_session=m)

    # Create a chain with requirements
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that writes professional emails."),
            ("human", "Write an email to {name} about: {topic}"),
        ]
    )

    # Bind model options with requirements
    model_with_requirements = chat_model.bind(
        model_options={
            "requirements": [
                req("The email should have a salutation"),
                req("The email should be professional and concise"),
            ],
            "strategy": RejectionSamplingStrategy(loop_budget=3),
        }
    )

    chain = prompt | model_with_requirements | StrOutputParser()

    result = chain.invoke({"name": "Dr. Smith", "topic": "upcoming conference presentation"})

    print(result)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Mellea Requirements and Strategy Examples")
    print("=" * 60 + "\n")

    # Run examples
    example_basic_requirements()
    example_requirements_with_strategy()
    example_custom_validation()
    example_with_langchain_chain()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
