# pytest: huggingface, llm, requires_heavy_ram
"""Example demonstrating GroundednessRequirement for grounded response validation.

This example shows how to use GroundednessRequirement to validate that
assistant responses are fully grounded by citations to retrieved documents.

The validator implements a sophisticated 4-step pipeline:
1. Citation Generation - Generate citations using the citations intrinsic
2. Citation Necessity - Identify which spans need citations (LLM judgment)
3. Citation Support - Assess support level for each span (LLM judgment)
4. Groundedness Output - Declare response grounded iff all spans needing
   citations are fully supported

Note: This example requires HuggingFace backend and access to the
ibm-granite/granite-4.0-micro model.
"""

import asyncio

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.rag import GroundednessRequirement


async def main():
    """Demonstrate GroundednessRequirement usage."""
    print("=" * 70)
    print("GroundednessRequirement Example")
    print("=" * 70)

    # Initialize HuggingFace backend
    print("\nInitializing HuggingFace backend...")
    backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

    # Create documents about Rupert Murdoch
    docs = [
        Document(
            doc_id="0",
            title="Murdoch Expansion",
            text=(
                "Keith Rupert Murdoch was born on 11 March 1931 in Melbourne, Australia. "
                "He began to direct his attention to acquisition and expansion, buying the "
                "troubled Sunday Times in Perth, Western Australia (1956) and over the next "
                "few years acquiring suburban and provincial newspapers in New South Wales, "
                "Queensland, Victoria and the Northern Territory, including the Sydney "
                "afternoon tabloid, The Daily Mirror (1960). "
                "Murdoch's first foray outside Australia involved the purchase of a "
                "controlling interest in the New Zealand daily The Dominion."
            ),
        ),
        Document(
            doc_id="1",
            title="Unrelated Document",
            text="This document has nothing to do with Rupert Murdoch.",
        ),
    ]

    # Example 1: Fully grounded response (all claims have citations)
    print("\n--- Example 1: Fully grounded response ---")
    response1 = (
        "Murdoch began his expansion in Australia by acquiring newspapers in Perth "
        "and other states. He then expanded to New Zealand with the New Zealand daily "
        "The Dominion."
    )

    ctx1 = ChatContext().add(
        Message("user", "How did Murdoch expand in Australia and New Zealand?")
    )
    ctx1 = ctx1.add(Message("assistant", response1, documents=docs))

    req1 = GroundednessRequirement(allow_partial_support=False)
    result1 = await req1.validate(backend, ctx1)

    print(f"Response: {response1}")
    print(f"Validation passed: {result1.as_bool()}")
    print(f"Reason:\n{result1.reason}")

    # Example 2: Partially grounded response (some claims lack citations)
    print("\n--- Example 2: Partially grounded response ---")
    response2 = (
        "Murdoch began his expansion in Perth and Queensland. "
        "He then moved to New Zealand and acquired The Dominion. "
        "Later he opened offices in Singapore and Hong Kong."  # This last claim has no citation
    )

    ctx2 = ChatContext().add(Message("user", "How did Murdoch expand geographically?"))
    ctx2 = ctx2.add(Message("assistant", response2, documents=docs))

    req2 = GroundednessRequirement(allow_partial_support=False)
    result2 = await req2.validate(backend, ctx2)

    print(f"Response: {response2}")
    print(f"Validation passed: {result2.as_bool()}")
    print(f"Reason:\n{result2.reason}")

    # Example 3: Same response with allow_partial_support=True
    print("\n--- Example 3: Same response with allow_partial_support=True ---")
    req3 = GroundednessRequirement(allow_partial_support=True)
    result3 = await req3.validate(backend, ctx2)

    print(f"Validation passed: {result3.as_bool()}")
    print(f"Reason:\n{result3.reason}")

    # Example 4: Response with no citations needed (conversational text)
    print("\n--- Example 4: Response with conversational text ---")
    response4 = (
        "I don't have enough information to answer this question fully. "
        "However, based on the documents provided, Murdoch did expand in Australia "
        "by acquiring newspapers in Perth and other states."
    )

    ctx4 = ChatContext().add(Message("user", "Tell me about Murdoch's expansion."))
    ctx4 = ctx4.add(Message("assistant", response4, documents=docs))

    req4 = GroundednessRequirement(allow_partial_support=False)
    result4 = await req4.validate(backend, ctx4)

    print(f"Response: {response4}")
    print(f"Validation passed: {result4.as_bool()}")
    print(f"Reason:\n{result4.reason}")

    # Example 5: Documents in constructor instead of message
    print("\n--- Example 5: Documents in constructor ---")
    response5 = (
        "Murdoch expanded in Australia by acquiring newspapers including "
        "the Sunday Times in Perth and the Daily Mirror in Sydney."
    )

    ctx5 = ChatContext().add(Message("user", "How did Murdoch expand in Australia?"))
    ctx5 = ctx5.add(Message("assistant", response5))

    req5 = GroundednessRequirement(documents=docs, allow_partial_support=False)
    result5 = await req5.validate(backend, ctx5)

    print(f"Response: {response5}")
    print(f"Validation passed: {result5.as_bool()}")
    print(f"Reason:\n{result5.reason}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
