# pytest: huggingface, requires_heavy_ram, llm, qualitative

"""Example using HallucinationRequirement with MelleaSession.

This example demonstrates how to use HallucinationRequirement in a MelleaSession
workflow for RAG applications. Since HallucinationRequirement validates documents
attached to assistant messages, we manually construct the context with documents.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/requirements/hallucination_session_example.py
```
"""

import mellea.stdlib.functional as mfuncs
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.requirements import HallucinationRequirement
from mellea.stdlib.session import MelleaSession

# Setup backend
backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

# Retrieved documents for RAG
documents = [
    Document(
        doc_id="1",
        text="The purple bumble fish is a rare species found in tropical waters. It has a distinctive yellow coloration.",
    ),
    Document(
        doc_id="2",
        text="Purple bumble fish typically grow to 15-20 cm in length and feed primarily on small crustaceans.",
    ),
    Document(
        doc_id="3",
        text="Conservation efforts have helped stabilize purple bumble fish populations in recent years.",
    ),
]

print("=" * 60)
print("Example 1: Direct validation workflow (recommended)")
print("=" * 60)

# For RAG workflows with HallucinationRequirement, construct the context
# with documents attached to assistant messages, then validate

from mellea.stdlib.context import ChatContext

# Create hallucination requirement
hallucination_req = HallucinationRequirement(
    threshold=0.5,
    max_hallucinated_ratio=0.0,  # Strict validation
)

# Create context with user question and assistant response with documents
context = (
    ChatContext()
    .add(Message("user", "What color are purple bumble fish?"))
    .add(
        Message(
            "assistant",
            "Purple bumble fish have distinctive yellow coloration.",
            documents=documents,
        )
    )
)

# Validate
validation_results = mfuncs.validate(
    reqs=[hallucination_req], context=context, backend=backend
)

print("Response: Purple bumble fish have distinctive yellow coloration.")
print(f"Validation passed: {validation_results[0].as_bool()}")
print(f"Validation reason: {validation_results[0].reason}")
if validation_results[0].score is not None:
    print(f"Faithfulness score: {validation_results[0].score:.2f}")

print("\n" + "=" * 60)
print("Example 2: Multi-document validation")
print("=" * 60)

# Test with a response that synthesizes information from multiple documents
multi_response = (
    "Purple bumble fish are rare tropical fish with yellow coloration. "
    "They grow to 15-20 cm and feed on small crustaceans. "
    "Conservation efforts have helped stabilize their populations."
)

context3 = (
    ChatContext()
    .add(Message("user", "Tell me about purple bumble fish."))
    .add(Message("assistant", multi_response, documents=documents))
)

validation_results3 = mfuncs.validate(
    reqs=[hallucination_req], context=context3, backend=backend
)

print(f"Response: {multi_response}")
print(f"Validation passed: {validation_results3[0].as_bool()}")
print(f"Validation reason: {validation_results3[0].reason}")
if validation_results3[0].score is not None:
    print(f"Faithfulness score: {validation_results3[0].score:.2f}")

print("\n" + "=" * 60)
print("Example 3: Detecting hallucinations")
print("=" * 60)

# Test with a response that includes hallucinated content
hallucinated_response = (
    "Purple bumble fish are rare tropical fish with yellow coloration. "
    "They grow to 15-20 cm and feed on small crustaceans. "
    "They are known to migrate thousands of miles each year."  # Hallucinated!
)

context4 = (
    ChatContext()
    .add(Message("user", "Tell me about purple bumble fish."))
    .add(Message("assistant", hallucinated_response, documents=documents))
)

validation_results4 = mfuncs.validate(
    reqs=[hallucination_req], context=context4, backend=backend
)

print(f"Response: {hallucinated_response}")
print(f"Validation passed: {validation_results4[0].as_bool()}")
print(f"Validation reason: {validation_results4[0].reason}")
if validation_results4[0].score is not None:
    print(f"Faithfulness score: {validation_results4[0].score:.2f}")

print("\n" + "=" * 60)
print("Example 4: Using session for generation, then validate")
print("=" * 60)

# You can use MelleaSession for generation, then manually validate
session = MelleaSession(backend=backend)

# Generate a response
session.ctx = session.ctx.add(Message("user", "What are purple bumble fish?"))
response = session.instruct(
    "Answer based on these facts: Purple bumble fish are tropical, yellow-colored fish that grow to 15-20 cm."
)

# Manually create context with documents for validation
validation_context = (
    ChatContext()
    .add(Message("user", "What are purple bumble fish?"))
    .add(Message("assistant", str(response), documents=documents))
)

validation_results = mfuncs.validate(
    reqs=[hallucination_req], context=validation_context, backend=backend
)

print(f"Generated response: {response}")
print(f"Validation passed: {validation_results[0].as_bool()}")
print(f"Validation reason: {validation_results[0].reason}")
if validation_results[0].score is not None:
    print(f"Faithfulness score: {validation_results[0].score:.2f}")

print("\n" + "=" * 60)
print("Examples complete!")
print("=" * 60)

