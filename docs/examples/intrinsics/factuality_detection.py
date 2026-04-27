# pytest: huggingface, e2e

"""Example usage of the factuality detection intrinsic.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/factuality_detection.py
```
"""

from mellea import start_backend, model_ids
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import guardian

user_text = "Is Ozzy Osbourne still alive?"
response_text = "Yes, Ozzy Osbourne is alive in 2025 and preparing for another world tour, continuing to amaze fans with his energy and resilience."

document = Document(
    # Context says Ozzy Osbourne is dead, but the response says he is alive.
    "Ozzy Osbourne passed away on July 22, 2025, at the age of 76 from a heart attack. "
    "He died at his home in Buckinghamshire, England, with contributing conditions "
    "including coronary artery disease and Parkinson's disease. His final "
    "performance took place earlier that month in Birmingham."
)

# Create the backend.
ctx, backend = start_backend("hf", model_id=model_ids.IBM_GRANITE_4_MICRO_3B, context_type="chat")
ctx = ctx.add(document).add(Message("user", user_text)).add(Message("assistant", response_text))

result = guardian.factuality_detection(ctx, backend)
print(f"Result of factuality detection: {result}")  # string "yes" or "no"
