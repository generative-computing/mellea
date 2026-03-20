# pytest: huggingface, requires_heavy_ram, llm, qualitative

"""RAG with instruct() and integrated hallucination validation.

This example shows how to integrate hallucination detection directly into
the instruct() workflow by using a helper function that manages document
attachment for validation.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/rag/rag_with_instruct_and_validation.py
```
"""

from mellea import start_session
from mellea.backends import model_ids
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.functional import validate
from mellea.stdlib.requirements import HallucinationRequirement
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Sample documents for RAG
docs = [
    "The purple bumble fish is a rare species found in tropical waters. It has a distinctive yellow coloration.",
    "Purple bumble fish typically grow to 15-20 cm in length and feed primarily on small crustaceans.",
    "Conservation efforts have helped stabilize purple bumble fish populations in recent years.",
]

print("=" * 60)
print("RAG with instruct() and Hallucination Validation")
print("=" * 60)


def instruct_with_rag_validation(
    session,
    prompt,
    documents,
    query=None,
    hallucination_req=None,
    user_variables=None,
    strategy=None,
):
    """Helper function to combine instruct() with hallucination validation.

    This function:
    1. Generates answer using instruct() with grounding_context
    2. Attaches documents to the generated message
    3. Validates for hallucinations
    4. Optionally retries if validation fails (with strategy)

    Args:
        session: MelleaSession instance
        prompt: Instruction prompt (can use {{query}} and {{doc0}}, {{doc1}}, etc.)
        documents: List of document strings
        query: Optional query string for user_variables
        hallucination_req: HallucinationRequirement instance
        user_variables: Additional user variables
        strategy: Sampling strategy for retries

    Returns:
        tuple: (answer_text, validation_result)
    """
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
    doc_objects = [Document(doc_id=str(i), text=doc) for i, doc in enumerate(documents)]

    # Get the context and attach documents to last message
    messages = session.ctx.as_list()
    if messages and isinstance(messages[-1], Message):
        # Create validation context with documents
        validation_ctx = ChatContext()
        for msg in messages[:-1]:
            validation_ctx = validation_ctx.add(msg)

        # Add last message with documents attached
        last_msg = messages[-1]
        msg_with_docs = Message(last_msg.role, last_msg.content, documents=doc_objects)
        validation_ctx = validation_ctx.add(msg_with_docs)

        # Validate
        validation_results = validate(
            reqs=[hallucination_req], context=validation_ctx, backend=session.backend
        )

        validation_result = validation_results[0]

        # Update session context with documents if validation passed
        if validation_result.as_bool():
            session.ctx = validation_ctx

        return str(answer.value), validation_result

    return str(answer.value), None


# Example 1: Basic usage
print("\nExample 1: Basic RAG with validation")
print("-" * 60)

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

# Example 2: With rejection sampling
print("\n" + "=" * 60)
print("Example 2: With Rejection Sampling")
print("-" * 60)

m2 = start_session(model_id=model_ids.IBM_GRANITE_4_MICRO_3B)

# Note: For true rejection sampling integration, you would need to modify
# the sampling strategy to handle document attachment. This example shows
# the manual approach.

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

# Example 3: Lenient validation
print("\n" + "=" * 60)
print("Example 3: Lenient Validation")
print("-" * 60)

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

print("\n" + "=" * 60)
print("Examples complete!")
print("=" * 60)
