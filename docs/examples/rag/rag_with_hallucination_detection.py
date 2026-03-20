# pytest: huggingface, requires_heavy_ram, llm, qualitative

"""RAG example with hallucination detection using HallucinationRequirement.

This example demonstrates how to integrate hallucination detection into a RAG
pipeline using Mellea's HallucinationRequirement.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/rag/rag_with_hallucination_detection.py
```
"""

from mellea import start_session
from mellea.backends import model_ids
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.functional import validate
from mellea.stdlib.requirements import HallucinationRequirement

# Sample documents for RAG
docs = [
    "The purple bumble fish is a rare species found in tropical waters. It has a distinctive yellow coloration.",
    "Purple bumble fish typically grow to 15-20 cm in length and feed primarily on small crustaceans.",
    "Conservation efforts have helped stabilize purple bumble fish populations in recent years.",
]

print("=" * 60)
print("RAG with Hallucination Detection Example")
print("=" * 60)

# Create session
m = start_session(model_id=model_ids.IBM_GRANITE_4_MICRO_3B)

# User query
query = "What do we know about purple bumble fish?"

# Step 1: Generate answer using RAG pattern with grounding_context
print("\nStep 1: Generating answer with grounded context...")
answer = m.instruct(
    "Based on the provided documents, answer the question: {{query}}",
    user_variables={"query": query},
    grounding_context={f"doc{i}": doc for i, doc in enumerate(docs)},
)

print(f"Generated answer: {answer.value}")

# Step 2: Validate for hallucinations
print("\nStep 2: Validating answer for hallucinations...")

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
print("\n" + "=" * 60)
print("Example with Hallucinated Content")
print("=" * 60)

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
print("\n" + "=" * 60)
print("Complete RAG Pipeline with Validation")
print("=" * 60)


def rag_with_validation(session, query, documents, requirement):
    """Complete RAG pipeline with hallucination detection.

    Args:
        session: MelleaSession instance
        query: User question
        documents: List of document strings
        requirement: HallucinationRequirement instance

    Returns:
        tuple: (answer, validation_result)
    """
    # Generate answer
    answer = session.instruct(
        "Based on the provided documents, answer the question: {{query}}",
        user_variables={"query": query},
        grounding_context={f"doc{i}": doc for i, doc in enumerate(documents)},
    )

    # Prepare for validation
    doc_objects = [Document(doc_id=str(i), text=doc) for i, doc in enumerate(documents)]

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

print("\n" + "=" * 60)
print("Example complete!")
print("=" * 60)
