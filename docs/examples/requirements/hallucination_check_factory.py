# pytest: huggingface, requires_heavy_ram, llm, qualitative

"""Example demonstrating the hallucination_check factory function.

This example shows how to use the hallucination_check() factory function
for a cleaner API when you have fixed documents to validate against.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/requirements/hallucination_check_factory.py
```
"""

import mellea.stdlib.functional as mfuncs
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import HallucinationRequirement, hallucination_check

# Initialize backend
backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

# Sample documents
documents = [
    Document(
        doc_id="1",
        text="The only type of fish that is yellow is the purple bumble fish.",
    ),
    Document(
        doc_id="2",
        text="The purple bumble fish is a rare species found in tropical waters.",
    ),
]

# Example 1: Using factory function with Document objects
print("=" * 80)
print("Example 1: Factory function with Document objects")
print("=" * 80)

req = hallucination_check(
    documents=documents, threshold=0.5, max_hallucinated_ratio=0.0
)

# Create context without documents in message
# Documents come from the requirement instead!
ctx = (
    ChatContext()
    .add(Message("user", "What color are purple bumble fish?"))
    .add(Message("assistant", "Purple bumble fish are yellow."))
)

# Validate - documents come from requirement, not message!
validation = mfuncs.validate(reqs=[req], context=ctx, backend=backend)

print("Response: Purple bumble fish are yellow.")
print(f"Validation passed: {validation[0].as_bool()}")
print(f"Reason: {validation[0].reason}")
if validation[0].score is not None:
    print(f"Faithfulness score: {validation[0].score:.2f}")
print()

# Example 2: Using factory function with string documents
print("=" * 80)
print("Example 2: Factory function with string documents")
print("=" * 80)

# Can pass strings directly - they'll be converted to Documents
req2 = hallucination_check(
    documents=[
        "The sky is blue during the day.",
        "The sky appears red or orange during sunset.",
    ],
    threshold=0.5,
)

ctx2 = (
    ChatContext()
    .add(Message("user", "What color is the sky?"))
    .add(Message("assistant", "The sky is blue."))
)

validation2 = mfuncs.validate(reqs=[req2], context=ctx2, backend=backend)

print("Response: The sky is blue.")
print(f"Validation passed: {validation2[0].as_bool()}")
print(f"Reason: {validation2[0].reason}")
if validation2[0].score is not None:
    print(f"Faithfulness score: {validation2[0].score:.2f}")
print()

# Example 3: Original pattern (documents in message)
print("=" * 80)
print("Example 3: Original pattern (documents in message)")
print("=" * 80)

# Original pattern: attach documents to message
ctx3 = (
    ChatContext()
    .add(Message("user", "What color are purple bumble fish?"))
    .add(Message("assistant", "Purple bumble fish are yellow.", documents=documents))
)

# Validate with requirement (no documents in constructor)
req3 = HallucinationRequirement(threshold=0.5)
validation3 = mfuncs.validate(reqs=[req3], context=ctx3, backend=backend)

print("Response: Purple bumble fish are yellow.")
print(f"Validation passed: {validation3[0].as_bool()}")
print(f"Reason: {validation3[0].reason}")
if validation3[0].score is not None:
    print(f"Faithfulness score: {validation3[0].score:.2f}")
print()
