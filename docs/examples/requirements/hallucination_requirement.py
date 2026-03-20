# pytest: huggingface, requires_heavy_ram, llm, qualitative

"""Example usage of HallucinationRequirement for RAG validation.

This example demonstrates how to use the HallucinationRequirement class
to validate that RAG responses are faithful to retrieved documents.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/requirements/hallucination_requirement.py
```
"""

import mellea.stdlib.functional as mfuncs
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import HallucinationRequirement

# Setup backend
backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

# Retrieved documents for single-document examples
single_document = [
    Document(
        doc_id="1",
        text="The only type of fish that is yellow is the purple bumble fish.",
    )
]

# Retrieved documents for multi-document examples
multi_documents = [
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
print("Example 1: Validating a faithful response (single document)")
print("=" * 60)

# Create context with a faithful response
context1 = (
    ChatContext()
    .add(Message("user", "What color are purple bumble fish?"))
    .add(
        Message(
            "assistant", "Purple bumble fish are yellow.", documents=single_document
        )
    )
)

# Create hallucination requirement with strict threshold
hallucination_check = HallucinationRequirement(
    threshold=0.5,
    max_hallucinated_ratio=0.0,  # Zero tolerance for hallucinations
)

# Validate the response
validation_results = mfuncs.validate(
    reqs=[hallucination_check], context=context1, backend=backend
)

print("Response: Purple bumble fish are yellow.")
print(f"Validation passed: {validation_results[0].as_bool()}")
print(f"Validation reason: {validation_results[0].reason}")
if validation_results[0].score is not None:
    print(f"Faithfulness score: {validation_results[0].score:.2f}")

# Example 2: Validating a hallucinated response
print("\n" + "=" * 60)
print("Example 2: Validating a hallucinated response")
print("=" * 60)

# Create context with a hallucinated response
context2 = (
    ChatContext()
    .add(Message("user", "What color are green bumble fish?"))
    .add(
        Message(
            "assistant", "Green bumble fish are also yellow.", documents=single_document
        )
    )
)

# Validate the hallucinated response
validation_results2 = mfuncs.validate(
    reqs=[hallucination_check], context=context2, backend=backend
)

print("Response: Green bumble fish are also yellow.")
print(f"Validation passed: {validation_results2[0].as_bool()}")
print(f"Validation reason: {validation_results2[0].reason}")
if validation_results2[0].score is not None:
    print(f"Faithfulness score: {validation_results2[0].score:.2f}")

# Example 3: Lenient validation
print("\n" + "=" * 60)
print("Example 3: Lenient validation (allow some hallucination)")
print("=" * 60)

lenient_req = HallucinationRequirement(
    threshold=0.5,
    max_hallucinated_ratio=0.3,  # Allow up to 30% hallucinated content
)

# Test with a response that has some hallucination
test_response = (
    "Purple bumble fish are yellow. Green bumble fish are also yellow. "
    "They live in the ocean and eat plankton."
)
context3 = (
    ChatContext()
    .add(Message("user", "Tell me about fish."))
    .add(Message("assistant", test_response, documents=single_document))
)

lenient_results = mfuncs.validate(reqs=[lenient_req], context=context3, backend=backend)

print(f"Response: {test_response}")
print(f"Lenient validation passed: {lenient_results[0].as_bool()}")
print(f"Validation reason: {lenient_results[0].reason}")
if lenient_results[0].score is not None:
    print(f"Faithfulness score: {lenient_results[0].score:.2f}")

# Example 4: Multi-document faithful response
print("\n" + "=" * 60)
print("Example 4: Multi-document faithful response")
print("=" * 60)

multi_response = (
    "Purple bumble fish are rare tropical fish with yellow coloration. "
    "They grow to 15-20 cm and feed on small crustaceans. "
    "Conservation efforts have helped stabilize their populations."
)
context4 = (
    ChatContext()
    .add(Message("user", "Tell me about purple bumble fish."))
    .add(Message("assistant", multi_response, documents=multi_documents))
)

multi_results = mfuncs.validate(
    reqs=[hallucination_check], context=context4, backend=backend
)

print(f"Response: {multi_response}")
print(f"Validation passed: {multi_results[0].as_bool()}")
print(f"Validation reason: {multi_results[0].reason}")
if multi_results[0].score is not None:
    print(f"Faithfulness score: {multi_results[0].score:.2f}")

# Example 5: Multi-document with partial hallucination
print("\n" + "=" * 60)
print("Example 5: Multi-document with partial hallucination")
print("=" * 60)

partial_hallucination = (
    "Purple bumble fish are rare tropical fish with yellow coloration. "
    "They grow to 15-20 cm and feed on small crustaceans. "
    "They are known to migrate thousands of miles each year."  # Hallucinated
)
context5 = (
    ChatContext()
    .add(Message("user", "Tell me about purple bumble fish."))
    .add(Message("assistant", partial_hallucination, documents=multi_documents))
)

partial_results = mfuncs.validate(
    reqs=[hallucination_check], context=context5, backend=backend
)

print(f"Response: {partial_hallucination}")
print(f"Validation passed: {partial_results[0].as_bool()}")
print(f"Validation reason: {partial_results[0].reason}")
if partial_results[0].score is not None:
    print(f"Faithfulness score: {partial_results[0].score:.2f}")

# Example 6: Multi-document with lenient validation
print("\n" + "=" * 60)
print("Example 6: Multi-document with lenient validation")
print("=" * 60)

context6 = (
    ChatContext()
    .add(Message("user", "Tell me about purple bumble fish."))
    .add(Message("assistant", partial_hallucination, documents=multi_documents))
)

lenient_multi_results = mfuncs.validate(
    reqs=[lenient_req], context=context6, backend=backend
)

print(f"Response: {partial_hallucination}")
print(f"Lenient validation passed: {lenient_multi_results[0].as_bool()}")
print(f"Validation reason: {lenient_multi_results[0].reason}")
if lenient_multi_results[0].score is not None:
    print(f"Faithfulness score: {lenient_multi_results[0].score:.2f}")

print("\n" + "=" * 60)
print("Examples complete!")
print("=" * 60)
