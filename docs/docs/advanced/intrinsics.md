---
title: "Adapter functions"
description: "Adapter-accelerated RAG quality checks using LoRA/aLoRA adapters with Granite models."
# diataxis: how-to
---

**Prerequisites:** use `uv sync --extra hf` for runtime LoRA/aLoRA adapter
functions and local [Granite Switch](/reference/glossary#granite-switch)
checkpoints. Both local paths require a GPU or Apple Silicon Mac. An
OpenAIBackend using a Granite Switch model served via vLLM uses
`uv sync --extra switch` when it downloads embedded adapter metadata.

Adapter functions are adapter-accelerated operations for RAG quality checks. They use
LoRA/aLoRA adapters loaded directly into the Hugging Face backend — faster and more
reliable than prompting a general-purpose model for these specialized micro-tasks.

> **Backend note:** Adapter functions work with two backends:
>
> - **LocalHFBackend** — loads LoRA/aLoRA adapters from the catalog at runtime.
>   A local Granite Switch checkpoint can instead use
>   `load_embedded_adapters=True`; install `mellea[hf]` first. Only
>   adapter functions embedded in the checkpoint are then available. Requires a
>   GPU or Apple Silicon Mac.
> - **OpenAIBackend** — uses a Granite Switch model served via vLLM with
>   `load_embedded_adapters=True`. Only adapter functions embedded in the model are
>   available — check the model's `adapter_index.json` for the list.
>   See `docs/docs/examples/granite-switch/README.md`
>
> Adapter functions do not work with Ollama or other remote backends.

Set up the backend once and reuse it across adapter function calls:

```python
# Requires: mellea[hf]
# Returns: LocalHFBackend
from mellea.backends.huggingface import LocalHFBackend

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
```

## Use a local Granite Switch checkpoint

Granite Switch checkpoints contain their adapter functions already. Pass the
checkpoint to `LocalHFBackend` with `load_embedded_adapters=True`; existing
helper functions such as `rag.check_answerability()` work unchanged.

```python
# Requires: mellea[hf]
# Returns: LocalHFBackend
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW

backend = LocalHFBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW,
    load_embedded_adapters=True,
)
```

Only adapter functions listed in the checkpoint's `adapter_index.json` are
available. Mellea warns once if Granite Switch's installed package metadata
does not yet include Mellea's resolved Transformers version; the warning
disappears after Granite Switch publishes compatible metadata.

Or, with a Granite Switch model via the OpenAI backend:

```python
from mellea.backends.openai import OpenAIBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW
from mellea.formatters import TemplateFormatter

backend = OpenAIBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name,
    formatter=TemplateFormatter(model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name),
    base_url="http://localhost:8000/v1",  # vLLM server
    api_key="EMPTY",
    load_embedded_adapters=True,
)
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

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
context = ChatContext().add(Message("assistant", "Hello! How can I help you?"))
question = "What is the square root of 4?"

docs_answerable = [Document("The square root of 4 is 2.")]
docs_not_answerable = [Document("The square root of 8 is approximately 2.83.")]

print(rag.check_answerability(question, docs_answerable, context, backend))   # True
print(rag.check_answerability(question, docs_not_answerable, context, backend))  # False
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

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
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

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
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

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
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

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
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

## Direct adapter function usage

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

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")

# Register an adapter by task name
req_adapter = CustomIntrinsicAdapter(
    "requirement-check",
    base_model_name=backend.base_model_name,
)
backend.add_adapter(req_adapter)

ctx = ChatContext()
ctx = ctx.add(Message("user", "Hi, can you help me?"))
ctx = ctx.add(Message("assistant", "Yes! What can I help with?"))

out, _ = mfuncs.act(
    Intrinsic(
        "requirement-check",
        intrinsic_kwargs={"requirement": "The assistant is helpful."},
    ),
    ctx,
    backend,
)
print(out)  # {"requirement_check": {"score": 1.0}}
```

The `Intrinsic` component loads aLoRA adapters (falling back to LoRA) by task name.
For OpenAI backends with Granite Switch, adapters are loaded from the model's
Hugging Face repository configuration instead of the adapter function catalog.
Output format is task-specific — `requirement-check` returns `{"requirement_check": {"score": <float>}}`.

## Composable adapter construction (advanced)

> **Advanced:** `Adapter` composes an `Identity`, an `IOContract`, and a
> weights binding into a single, inspectable object. It's scaffolding for a
> future backend-integration surface (Epic #929) — today, neither backend
> accepts a composed `Adapter` directly: `LocalHFBackend.add_adapter` takes a
> `LocalFileBinding` or the `LocalHFAdapter` shim, while
> `OpenAIBackend.add_adapter` takes only the deprecated
> `EmbeddedIntrinsicAdapter` shim, which builds an `EmbeddedBinding`
> internally. The construction below is illustrative of the binding shapes;
> write a new backend integration against the bindings themselves.

Each weights binding models how its deployment turns an adapter on.
`LocalFileBinding` downloads and loads LoRA/aLoRA weights, so it exposes a
`prepare`/`activate`/`deactivate`/`release` lifecycle:

```python
# Requires: mellea[hf]
from mellea.backends.adapters import Adapter, EmbeddedBinding, Identity, IOContract, LocalFileBinding
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.openai import OpenAIBackend
from mellea.core import Component


class AnswerabilityContract(IOContract):
    def build_prompt(self, **kwargs: object) -> Component:
        raise NotImplementedError  # request formatting lands with #1516

    def parse(self, raw: str) -> dict[str, object]:
        import json

        return json.loads(raw)


# LocalFile/PEFT reality — LocalHFBackend downloads and loads the weights.
hf_backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
hf_binding = LocalFileBinding.from_catalog("answerability")
hf_binding.bind_backend(hf_backend)
# hf_binding.prepare() downloads the weights and loads them into hf_backend.
# adapter_type must match the binding — from_catalog loads the first
# catalog-listed adapter type, which is LoRA for answerability.
hf_adapter = Adapter(
    identity=Identity(name="answerability", adapter_type="lora"),
    io_contract=AnswerabilityContract(),
    weights=hf_binding,
)
```

`EmbeddedBinding` has no weights to manage — the adapter is already part of
the served base model — so it exposes a single method, `apply_activation`,
that edits the outgoing request instead of a lifecycle:

```python
switch_backend = OpenAIBackend(
    model_id="granite-switch",
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)
switch_adapter = Adapter(
    identity=Identity(name="answerability", adapter_type="alora"),
    io_contract=AnswerabilityContract(),
    weights=EmbeddedBinding.from_base_model(switch_backend),
)
```

Weights-binding support by backend today — this tracks the binding
implementations, not whether a composed `Adapter` can be registered directly:

| Backend | `LocalFileBinding` (LocalFile/PEFT) | `EmbeddedBinding` (Embedded/Granite Switch) | `ServerMediatedBinding` |
| --- | --- | --- | --- |
| `LocalHFBackend` | ✅ shipping — `add_adapter` accepts a `LocalFileBinding` directly | ✅ shipping — `load_embedded_adapters=True`, via the deprecated `EmbeddedIntrinsicAdapter` shim | — |
| `OpenAIBackend` | — | ✅ shipping, via the deprecated `EmbeddedIntrinsicAdapter` shim above, which builds an `EmbeddedBinding` internally | — |

`ServerMediatedBinding` has no backend implementation yet — see discussion #1486.

---

## Guardian adapter functions

Safety and factuality checks use a separate set of Guardian-specific adapter functions:
`guardian_check()`, `policy_guardrails()`, `factuality_detection()`, and
`factuality_correction()`. These are documented in the
[Safety Guardrails](../how-to/safety-guardrails) how-to guide.
