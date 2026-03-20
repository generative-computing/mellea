"""Example: RAG with instruct() and hallucination_check factory.

This example demonstrates using the hallucination_check() factory function
with instruct() and sampling strategies for automatic validation and repair.

# pytest: ollama, llm, huggingface
"""

import asyncio

from mellea.backends.hf import HuggingFaceBackend

from mellea.stdlib.components import Document
from mellea.stdlib.functional import instruct
from mellea.stdlib.requirements import hallucination_check
from mellea.stdlib.sampling import RejectionSamplingStrategy


async def main():
    """Demonstrate RAG with instruct() and hallucination_check factory."""
    # Initialize backend
    backend = HuggingFaceBackend(
        "ibm-granite/granite-3.0-2b-instruct", device_map="auto"
    )

    # Sample documents for RAG
    documents = [
        Document(
            doc_id="1",
            text="The only type of fish that is yellow is the purple bumble fish.",
        ),
        Document(
            doc_id="2",
            text="The purple bumble fish is a rare species found in tropical waters.",
        ),
        Document(
            doc_id="3",
            text="Purple bumble fish typically grow to 6-8 inches in length.",
        ),
    ]

    # Example 1: Basic usage with rejection sampling
    print("=" * 80)
    print("Example 1: instruct() with hallucination_check and rejection sampling")
    print("=" * 80)

    # Create requirement with factory function
    req = hallucination_check(
        documents=documents,
        threshold=0.5,
        max_hallucinated_ratio=0.0,  # Strict: no hallucinations allowed
    )

    # Use with instruct() - automatic validation and retry
    result = await instruct(
        """Based on the provided documents, answer the following question.

Question: What color are purple bumble fish?

Answer:""",
        backend=backend,
        requirements=[req],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )

    print(f"Response: {result}")
    print()

    # Example 2: With grounding context for prompt templating
    print("=" * 80)
    print("Example 2: Using grounding_context with hallucination_check")
    print("=" * 80)

    query = "How big do purple bumble fish grow?"

    # Create requirement with documents
    req2 = hallucination_check(
        documents=documents,
        threshold=0.5,
        max_hallucinated_ratio=0.1,  # Allow up to 10% hallucination
    )

    # Use grounding_context for prompt variables
    result2 = await instruct(
        """Based on the provided documents, answer: {{query}}

Answer:""",
        backend=backend,
        grounding_context={"query": query},
        requirements=[req2],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )

    print(f"Query: {query}")
    print(f"Response: {result2}")
    print()

    # Example 3: Multiple requirements including hallucination check
    print("=" * 80)
    print("Example 3: Combining hallucination_check with other requirements")
    print("=" * 80)

    from mellea.stdlib.requirements import Requirement

    # Multiple requirements
    requirements = [
        hallucination_check(documents=documents, threshold=0.5),
        Requirement("Response must be concise (under 50 words)"),
        Requirement("Response must be in complete sentences"),
    ]

    result3 = await instruct(
        "Describe purple bumble fish based on the documents.",
        backend=backend,
        requirements=requirements,
        strategy=RejectionSamplingStrategy(loop_budget=5),
    )

    print(f"Response: {result3}")
    print()

    # Example 4: Lenient hallucination tolerance
    print("=" * 80)
    print("Example 4: Lenient hallucination tolerance")
    print("=" * 80)

    # Allow some hallucination (useful for creative responses)
    lenient_req = hallucination_check(
        documents=documents,
        threshold=0.3,  # Lower threshold
        max_hallucinated_ratio=0.3,  # Allow up to 30% hallucination
    )

    result4 = await instruct(
        """Based on the documents, write a creative description of purple bumble fish.

Description:""",
        backend=backend,
        requirements=[lenient_req],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )

    print(f"Response: {result4}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
