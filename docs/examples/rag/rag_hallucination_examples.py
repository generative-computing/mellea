# pytest: huggingface, requires_heavy_ram, llm, qualitative

"""RAG hallucination detection examples - executes all three RAG examples sequentially.

This script combines and executes three RAG examples:
1. RAG with hallucination detection using HallucinationRequirement
2. RAG with instruct() and hallucination_check factory
3. RAG with instruct() and integrated validation

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/rag/rag_hallucination_examples.py
```
"""

import asyncio

from mellea import start_session
from mellea.backends import model_ids
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.functional import instruct, validate
from mellea.stdlib.requirements import (
    HallucinationRequirement,
    Requirement,
    hallucination_check,
)
from mellea.stdlib.sampling import RejectionSamplingStrategy


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subsection_header(title):
    """Print a formatted subsection header."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# ============================================================================
# EXAMPLE 1: RAG with Hallucination Detection
# ============================================================================


def example_1_rag_with_hallucination_detection():
    """RAG example with hallucination detection using HallucinationRequirement."""
    print_section_header("EXAMPLE 1: RAG with Hallucination Detection")

    # Sample documents for RAG
    docs = [
        "The purple bumble fish is a rare species found in tropical waters. It has a distinctive yellow coloration.",
        "Purple bumble fish typically grow to 15-20 cm in length and feed primarily on small crustaceans.",
        "Conservation efforts have helped stabilize purple bumble fish populations in recent years.",
    ]

    # Create session
    m = start_session(model_id=model_ids.IBM_GRANITE_4_MICRO_3B)

    # User query
    query = "What do we know about purple bumble fish?"

    # Step 1: Generate answer using RAG pattern with grounding_context
    print_subsection_header("Step 1: Generating answer with grounded context")
    answer = m.instruct(
        "Based on the provided documents, answer the question: {{query}}",
        user_variables={"query": query},
        grounding_context={f"doc{i}": doc for i, doc in enumerate(docs)},
    )

    print(f"Generated answer: {answer.value}")

    # Step 2: Validate for hallucinations
    print_subsection_header("Step 2: Validating answer for hallucinations")

    # Create Document objects for validation
    doc_objects = [Document(doc_id=str(i), text=doc) for i, doc in enumerate(docs)]

    # Build validation context with documents attached to assistant message
    validation_context = (
        ChatContext()
        .add(Message("user", query))
        .add(Message("assistant", str(answer.value), documents=doc_objects))
    )

    # Create hallucination requirement
    hallucination_req = HallucinationRequirement(
        threshold=0.5,
        max_hallucinated_ratio=0.0,  # Strict: no hallucinations allowed
    )

    # Validate
    validation_results = validate(
        reqs=[hallucination_req], context=validation_context, backend=m.backend
    )

    print(f"Validation passed: {validation_results[0].as_bool()}")
    print(f"Validation reason: {validation_results[0].reason}")
    if validation_results[0].score is not None:
        print(f"Faithfulness score: {validation_results[0].score:.2f}")

    # Step 3: Example with potential hallucination
    print_subsection_header("Step 3: Example with Hallucinated Content")

    # Manually create a response with hallucination for demonstration
    hallucinated_answer = (
        "Purple bumble fish are rare tropical fish with yellow coloration. "
        "They grow to 15-20 cm and feed on small crustaceans. "
        "They are known to migrate thousands of miles each year."  # Hallucinated!
    )

    validation_context2 = (
        ChatContext()
        .add(Message("user", query))
        .add(Message("assistant", hallucinated_answer, documents=doc_objects))
    )

    validation_results2 = validate(
        reqs=[hallucination_req], context=validation_context2, backend=m.backend
    )

    print(f"Response: {hallucinated_answer}")
    print(f"Validation passed: {validation_results2[0].as_bool()}")
    print(f"Validation reason: {validation_results2[0].reason}")
    if validation_results2[0].score is not None:
        print(f"Faithfulness score: {validation_results2[0].score:.2f}")

    # Step 4: Complete RAG pipeline with validation
    print_subsection_header("Step 4: Complete RAG Pipeline with Validation")

    def rag_with_validation(session, query, documents, requirement):
        """Complete RAG pipeline with hallucination detection."""
        # Generate answer
        answer = session.instruct(
            "Based on the provided documents, answer the question: {{query}}",
            user_variables={"query": query},
            grounding_context={f"doc{i}": doc for i, doc in enumerate(documents)},
        )

        # Prepare for validation
        doc_objects = [
            Document(doc_id=str(i), text=doc) for i, doc in enumerate(documents)
        ]

        validation_context = (
            ChatContext()
            .add(Message("user", query))
            .add(Message("assistant", str(answer.value), documents=doc_objects))
        )

        # Validate
        validation_results = validate(
            reqs=[requirement], context=validation_context, backend=session.backend
        )

        return answer.value, validation_results[0]

    # Use the pipeline
    query2 = "How big do purple bumble fish grow?"
    answer, validation = rag_with_validation(m, query2, docs, hallucination_req)

    print(f"Query: {query2}")
    print(f"Answer: {answer}")
    print(f"Validated: {validation.as_bool()}")
    if validation.score is not None:
        print(f"Faithfulness score: {validation.score:.2f}")

    print("\nExample 1 complete!")


# ============================================================================
# EXAMPLE 2: RAG with instruct() and hallucination_check factory
# ============================================================================


async def example_2_rag_with_instruct_and_factory():
    """RAG with instruct() and hallucination_check factory."""
    print_section_header(
        "EXAMPLE 2: RAG with instruct() and hallucination_check factory"
    )

    # Initialize backend
    backend = LocalHFBackend("ibm-granite/granite-3.0-2b-instruct")

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

    # Example 2.1: Basic usage with rejection sampling
    print_subsection_header(
        "Example 2.1: instruct() with hallucination_check and rejection sampling"
    )

    from mellea.stdlib.context import SimpleContext

    # Create requirement with factory function
    req = hallucination_check(
        documents=documents,
        threshold=0.5,
        max_hallucinated_ratio=0.0,  # Strict: no hallucinations allowed
    )

    # Use with instruct() - automatic validation and retry
    result, _ = instruct(
        """Based on the provided documents, answer the following question.

Question: What color are purple bumble fish?

Answer:""",
        SimpleContext(),
        backend,
        requirements=[req],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )

    print(f"Response: {result.value}")

    # Example 2.2: With grounding context for prompt templating
    print_subsection_header(
        "Example 2.2: Using grounding_context with hallucination_check"
    )

    query = "How big do purple bumble fish grow?"

    # Create requirement with documents
    req2 = hallucination_check(
        documents=documents,
        threshold=0.5,
        max_hallucinated_ratio=0.1,  # Allow up to 10% hallucination
    )

    # Use grounding_context for prompt variables
    result2, _ = instruct(
        """Based on the provided documents, answer: {{query}}

Answer:""",
        SimpleContext(),
        backend,
        grounding_context={"query": query},
        requirements=[req2],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )

    print(f"Query: {query}")
    print(f"Response: {result2.value}")

    # Example 2.3: Multiple requirements including hallucination check
    print_subsection_header(
        "Example 2.3: Combining hallucination_check with other requirements"
    )

    # Multiple requirements
    requirements = [
        hallucination_check(documents=documents, threshold=0.5),
        Requirement("Response must be concise (under 50 words)"),
        Requirement("Response must be in complete sentences"),
    ]

    result3, _ = instruct(
        "Describe purple bumble fish based on the documents.",
        SimpleContext(),
        backend,
        requirements=requirements,
        strategy=RejectionSamplingStrategy(loop_budget=5),
    )

    print(f"Response: {result3.value}")

    # Example 2.4: Lenient hallucination tolerance
    print_subsection_header("Example 2.4: Lenient hallucination tolerance")

    # Allow some hallucination (useful for creative responses)
    lenient_req = hallucination_check(
        documents=documents,
        threshold=0.3,  # Lower threshold
        max_hallucinated_ratio=0.3,  # Allow up to 30% hallucination
    )

    result4, _ = instruct(
        """Based on the documents, write a creative description of purple bumble fish.

Description:""",
        SimpleContext(),
        backend,
        requirements=[lenient_req],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )

    print(f"Response: {result4.value}")
    print("\nExample 2 complete!")


# ============================================================================
# EXAMPLE 3: RAG with instruct() and integrated validation
# ============================================================================


def example_3_rag_with_instruct_and_validation():
    """RAG with instruct() and integrated hallucination validation."""
    print_section_header("EXAMPLE 3: RAG with instruct() and integrated validation")

    # Sample documents for RAG
    docs = [
        "The purple bumble fish is a rare species found in tropical waters. It has a distinctive yellow coloration.",
        "Purple bumble fish typically grow to 15-20 cm in length and feed primarily on small crustaceans.",
        "Conservation efforts have helped stabilize purple bumble fish populations in recent years.",
    ]

    def instruct_with_rag_validation(
        session,
        prompt,
        documents,
        query=None,
        hallucination_req=None,
        user_variables=None,
        strategy=None,
    ):
        """Helper function to combine instruct() with hallucination validation."""
        # Prepare user variables
        vars_dict = user_variables or {}
        if query:
            vars_dict["query"] = query

        # Generate answer with grounding context
        answer = session.instruct(
            prompt,
            user_variables=vars_dict,
            grounding_context={f"doc{i}": doc for i, doc in enumerate(documents)},
            strategy=None,  # Don't use strategy yet, we'll handle validation manually
        )

        # If no validation required, return early
        if hallucination_req is None:
            return str(answer.value), None

        # Prepare documents for validation
        doc_objects = [
            Document(doc_id=str(i), text=doc) for i, doc in enumerate(documents)
        ]

        # Get the context and attach documents to last message
        messages = session.ctx.as_list()
        if messages and isinstance(messages[-1], Message):
            # Create validation context with documents
            validation_ctx = ChatContext()
            for msg in messages[:-1]:
                validation_ctx = validation_ctx.add(msg)

            # Add last message with documents attached
            last_msg = messages[-1]
            msg_with_docs = Message(
                last_msg.role, last_msg.content, documents=doc_objects
            )
            validation_ctx = validation_ctx.add(msg_with_docs)

            # Validate
            validation_results = validate(
                reqs=[hallucination_req],
                context=validation_ctx,
                backend=session.backend,
            )

            validation_result = validation_results[0]

            # Update session context with documents if validation passed
            if validation_result.as_bool():
                session.ctx = validation_ctx

            return str(answer.value), validation_result

        return str(answer.value), None

    # Example 3.1: Basic usage
    print_subsection_header("Example 3.1: Basic RAG with validation")

    m = start_session(model_id=model_ids.IBM_GRANITE_4_MICRO_3B)

    hallucination_req = HallucinationRequirement(
        threshold=0.5,
        max_hallucinated_ratio=0.0,  # Strict
    )

    query = "What do we know about purple bumble fish?"
    answer, validation = instruct_with_rag_validation(
        session=m,
        prompt="Based on the provided documents, answer the question: {{query}}",
        documents=docs,
        query=query,
        hallucination_req=hallucination_req,
    )

    print(f"Query: {query}")
    print(f"Answer: {answer}")
    if validation:
        print(f"Validated: {validation.as_bool()}")
        print(f"Reason: {validation.reason}")
        if validation.score is not None:
            print(f"Faithfulness score: {validation.score:.2f}")

    # Example 3.2: With rejection sampling
    print_subsection_header("Example 3.2: With Rejection Sampling")

    m2 = start_session(model_id=model_ids.IBM_GRANITE_4_MICRO_3B)

    query2 = "How big do purple bumble fish grow?"
    max_attempts = 3

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1}/{max_attempts}")

        answer2, validation2 = instruct_with_rag_validation(
            session=m2,
            prompt="Based on the provided documents, answer: {{query}}",
            documents=docs,
            query=query2,
            hallucination_req=hallucination_req,
        )

        print(f"Answer: {answer2}")
        if validation2:
            print(f"Validated: {validation2.as_bool()}")
            if validation2.as_bool():
                print("✓ Validation passed!")
                break
            else:
                print(f"✗ Validation failed: {validation2.reason}")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                    m2.reset()  # Reset context for retry
        else:
            break

    # Example 3.3: Lenient validation
    print_subsection_header("Example 3.3: Lenient Validation")

    m3 = start_session(model_id=model_ids.IBM_GRANITE_4_MICRO_3B)

    lenient_req = HallucinationRequirement(
        threshold=0.5,
        max_hallucinated_ratio=0.3,  # Allow up to 30%
    )

    query3 = "Tell me everything about purple bumble fish"
    answer3, validation3 = instruct_with_rag_validation(
        session=m3,
        prompt="Based on the documents, provide a comprehensive answer to: {{query}}",
        documents=docs,
        query=query3,
        hallucination_req=lenient_req,
    )

    print(f"Query: {query3}")
    print(f"Answer: {answer3}")
    if validation3:
        print(f"Validated: {validation3.as_bool()}")
        print(f"Reason: {validation3.reason}")
        if validation3.score is not None:
            print(f"Faithfulness score: {validation3.score:.2f}")

    print("\nExample 3 complete!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """Execute all three RAG examples sequentially."""
    print("\n" + "=" * 80)
    print("COMBINED RAG EXAMPLES - EXECUTING ALL THREE EXAMPLES")
    print("=" * 80)

    # Execute Example 1
    example_1_rag_with_hallucination_detection()

    # Execute Example 2 (async)
    await example_2_rag_with_instruct_and_factory()

    # Execute Example 3
    example_3_rag_with_instruct_and_validation()

    # Final summary
    print_section_header("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("\nSummary:")
    print("✓ Example 1: RAG with Hallucination Detection")
    print("✓ Example 2: RAG with instruct() and hallucination_check factory")
    print("✓ Example 3: RAG with instruct() and integrated validation")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

# Made with Bob
