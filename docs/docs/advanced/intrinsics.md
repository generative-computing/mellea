---
title: "Intrinsics"
description: "Adapter-accelerated RAG quality checks using LoRA/aLoRA adapters with Granite models."
# diataxis: how-to
---

**Prerequisites:** `pip install "mellea[hf]"`, a GPU or Apple Silicon Mac recommended for
acceptable inference speed. All intrinsics require a `LocalHFBackend` with a
[Granite](https://huggingface.co/ibm-granite) model.

Intrinsics are adapter-accelerated operations for RAG quality checks. They use
LoRA/aLoRA adapters loaded directly into the HuggingFace backend — faster and more
reliable than prompting a general-purpose model for these specialized micro-tasks.

> **Backend note:** Intrinsics require `LocalHFBackend` with an IBM Granite model
> (e.g., `ibm-granite/granite-4.0-micro`). They do not work with Ollama, OpenAI, or
> other remote backends.

Set up the backend once and reuse it across intrinsic calls:

```python
# Requires: mellea[hf]
# Returns: LocalHFBackend
from mellea.backends.huggingface import LocalHFBackend

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
```

## Answerability

Check whether a set of retrieved documents can answer a given question:

```python
# Requires: mellea[hf]
# Returns: bool
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = ChatContext().add(Message("assistant", "Hello! How can I help you?"))
question = "What is the square root of 4?"

docs_answerable = [Document("The square root of 4 is 2.")]
docs_not_answerable = [Document("The square root of 8 is approximately 2.83.")]

print(rag.check_answerability(question, docs_answerable, context, backend))   # True
print(rag.check_answerability(question, docs_not_answerable, context, backend))  # False
```

## Context relevance

Assess whether a document is relevant to a question:

```python
# Requires: mellea[hf]
# Returns: float
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = ChatContext()
question = "Who is the CEO of Microsoft?"
document = Document(
    "Microsoft Corporation is an American multinational corporation "
    "headquartered in Redmond, Washington."
)

result = rag.check_context_relevance(question, document, context, backend)
print(result)  # False — the document does not mention the CEO
```

## Hallucination detection

Flag sentences in an assistant response that are not grounded in the source documents:

```python
# Requires: mellea[hf]
# Returns: list[str]
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = (
    ChatContext()
    .add(Message("assistant", "Hello! How can I help you?"))
    .add(Message("user", "Tell me about yellow fish."))
)

response = "Purple bumble fish are yellow. Green bumble fish are also yellow."
documents = [
    Document(doc_id="1", text="The only type of fish that is yellow is the purple bumble fish.")
]

result = rag.flag_hallucinated_content(response, documents, context, backend)
print(result)
# Flags "Green bumble fish are also yellow." as hallucinated
```

## Answer relevance rewriting

Rewrite a vague or incomplete answer to be more grounded in the source documents:

```python
# Requires: mellea[hf]
# Returns: str
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = ChatContext().add(Message("user", "Who attended the meeting?"))
documents = [
    Document("Meeting attendees: Alice, Bob, Carol."),
    Document("Meeting time: 9:00 am to 11:00 am."),
]
original = "Many people attended the meeting."

result = rag.rewrite_answer_for_relevance(original, documents, context, backend)
print(result)
# A more specific, grounded answer — output will vary
```

## Query rewriting

Rewrite an ambiguous user query using conversation history to improve retrieval:

```python
# Requires: mellea[hf]
# Returns: str
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = (
    ChatContext()
    .add(Message("assistant", "Welcome to pet questions!"))
    .add(Message("user", "I have two pets: a dog named Rex and a cat named Lucy."))
    .add(Message("assistant", "Rex spends a lot of time outdoors, and Lucy is always inside."))
    .add(Message("user", "Sounds good! Rex must love exploring outside."))
)
next_turn = "But is he more likely to get fleas because of that?"

result = rag.rewrite_question(next_turn, context, backend)
print(result)
# Resolves "he" to "Rex" and incorporates context about outdoor exposure
```

## Citations

Find supporting sentences in source documents for a given assistant response:

```python
# Requires: mellea[hf]
# Returns: dict
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = ChatContext().add(
    Message("user", "How did Murdoch expand in Australia versus New Zealand?")
)
response = (
    "Murdoch expanded in Australia and New Zealand by acquiring local newspapers. "
    "I do not have information about his expansion in New Zealand after purchasing "
    "The Dominion."
)
documents = [
    Document(doc_id="1", text="Keith Rupert Murdoch was born on 11 March 1931 in Melbourne..."),
    Document(doc_id="2", text="This document has nothing to do with Rupert Murdoch."),
]

result = rag.find_citations(response, documents, context, backend)
print(result)
# Maps each response sentence to supporting document sentences
```

## Direct intrinsic usage

> **Advanced:** For custom adapter tasks, use the `Intrinsic` component and
> `CustomIntrinsicAdapter` directly.

```python
# Requires: mellea[hf]
# Returns: dict
import mellea.stdlib.functional as mfuncs
from mellea.backends.adapters.adapter import CustomIntrinsicAdapter
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

# Register an adapter by task name
req_adapter = CustomIntrinsicAdapter(
    "requirement_check",
    base_model_name=backend.base_model_name,
)
backend.add_adapter(req_adapter)

ctx = ChatContext()
ctx = ctx.add(Message("user", "Hi, can you help me?"))
ctx = ctx.add(Message("assistant", "Yes! What can I help with?"))

out, _ = mfuncs.act(
    Intrinsic(
        "requirement_check",
        intrinsic_kwargs={"requirement": "The assistant is helpful."},
    ),
    ctx,
    backend,
)
print(out)  # {"requirement_likelihood": 1.0}
```

The `Intrinsic` component loads aLoRA adapters (falling back to LoRA) by task name.
Output format is task-specific — `requirement_check` returns a likelihood score.
