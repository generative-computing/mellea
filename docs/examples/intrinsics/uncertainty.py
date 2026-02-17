# pytest: huggingface, requires_heavy_ram, llm

"""Example usage of the uncertainty/certainty intrinsic.

Evaluates how certain the model is about its response to a user question.
The context should contain a user question followed by an assistant answer.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/uncertainty.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = (
    ChatContext()
    .add(Message("user", "What is the square root of 4?"))
    .add(Message("assistant", "The square root of 4 is 2."))
)

result = core.check_certainty(context, backend)
print(f"Certainty score: {result}")
