# pytest: huggingface, llm, requires_heavy_ram
"""Example demonstrating CitationRequirement for RAG workflows.

This example shows how to use CitationRequirement to validate that
assistant responses properly cite their sources in RAG workflows.

Note: This example requires HuggingFace backend and access to the
ibm-granite/granite-4.0-micro model.
"""

import asyncio

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.rag import CitationMode, CitationRequirement


async def main():
    """Demonstrate CitationRequirement usage."""
    print("=" * 70)
    print("CitationRequirement Example")
    print("=" * 70)

    # Initialize HuggingFace backend
    print("\nInitializing HuggingFace backend...")
    backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

    # Create documents
    docs = [
        Document(
            doc_id="doc1",
            title="Sky Facts",
            text="The sky appears blue during the day due to Rayleigh scattering.",
        ),
        Document(
            doc_id="doc2",
            title="Grass Facts",
            text="Grass is typically green because of chlorophyll in the leaves.",
        ),
    ]

    # Create a response that should have citations
    response = (
        "The sky appears blue during the day. "
        "Grass is green because it contains chlorophyll."
    )

    # Create context
    print("\nCreating context with user question and assistant response...")
    ctx = ChatContext().add(Message("user", "What colors are the sky and grass?"))
    ctx = ctx.add(Message("assistant", response, documents=docs))

    # Example 1: Documents in constructor
    print("\n--- Example 1: CitationRequirement with documents in constructor ---")
    req = CitationRequirement(min_citation_coverage=0.7, documents=docs)
    result = await req.validate(backend, ctx)

    print(f"Validation passed: {result.as_bool()}")
    print(f"Citation coverage score: {result.score:.2%}")
    if result.reason:
        reason_preview = (
            result.reason[:200] + "..." if len(result.reason) > 200 else result.reason
        )
        print(f"Reason: {reason_preview}")

    # Example 2: CLAIMS mode (default) - counts fraction of claims with citations
    print("\n--- Example 2: CLAIMS mode (default) - fraction of claims cited ---")
    req2 = CitationRequirement(
        min_citation_coverage=0.7, documents=docs, mode=CitationMode.CLAIMS
    )
    result2 = await req2.validate(backend, ctx)

    print(f"Validation passed: {result2.as_bool()}")
    print(f"Citation coverage score: {result2.score:.2%}")
    if result2.reason:
        reason_preview = (
            result2.reason[:200] + "..."
            if len(result2.reason) > 200
            else result2.reason
        )
        print(f"Reason: {reason_preview}")

    # Example 3: CHARACTERS mode - calculates character-based coverage
    print("\n--- Example 3: CHARACTERS mode - character-based coverage ---")
    req3 = CitationRequirement(
        min_citation_coverage=0.7, documents=docs, mode=CitationMode.CHARACTERS
    )
    result3 = await req3.validate(backend, ctx)

    print(f"Validation passed: {result3.as_bool()}")
    print(f"Citation coverage score: {result3.score:.2%}")
    if result3.reason:
        reason_preview = (
            result3.reason[:200] + "..."
            if len(result3.reason) > 200
            else result3.reason
        )
        print(f"Reason: {reason_preview}")

    # Example 4: Documents attached to message
    print("\n--- Example 4: Documents in message (not constructor) ---")
    ctx2 = ChatContext().add(Message("user", "Tell me about Mars."))
    ctx2 = ctx2.add(
        Message(
            "assistant",
            "Mars is the fourth planet from the Sun.",
            documents=[
                Document(doc_id="doc1", text="Mars is the fourth planet from the Sun.")
            ],
        )
    )

    req4 = CitationRequirement(min_citation_coverage=0.7)  # No documents in constructor
    result4 = await req4.validate(backend, ctx2)

    print(f"Validation passed: {result4.as_bool()}")
    print(f"Citation coverage score: {result4.score:.2%}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
