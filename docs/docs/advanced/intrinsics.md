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

> **Advanced:** For custom adapter tasks, compose an `Adapter` directly from
> an `Identity`, an output contract, and a weights binding.

```python
# Requires: mellea[hf]
# Returns: dict
import mellea.stdlib.functional as mfuncs
from mellea.backends.adapters import Adapter, Identity, LocalFileBinding, get_io_contract
from mellea.backends.adapters.catalog import AdapterType, fetch_intrinsic_metadata
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")

# Compose an adapter by task name — get_io_contract returns the catalog's
# declared contract (or a permissive fallback for a name outside it).
# requirement-check's catalog entry lists LoRA before aLoRA, so
# LocalFileBinding.from_catalog would pick LoRA — build the binding
# explicitly instead when you want the aLoRA variant specifically; identity
# and weights must agree, since nothing currently cross-checks them.
_metadata = fetch_intrinsic_metadata("requirement-check")
req_adapter = Adapter(
    identity=Identity(
        name="requirement-check",
        adapter_type="alora",
        capability=_metadata.effective_capability,
    ),
    io_contract=get_io_contract("requirement-check"),
    weights=LocalFileBinding(
        name="requirement-check",
        adapter_type=AdapterType.ALORA,
        repo_id=_metadata.repo_id,
        revision=_metadata.revision,
    ),
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

For a fully custom, non-catalog adapter — your own trained LoRA/aLoRA weights,
not one of the built-in adapter functions — see
[Adding a custom adapter function in 20 lines](../tutorials/07-custom-adapter-function.md).

## Composable adapter construction (advanced)

`Adapter` composes an `Identity`, an output contract (`IOContract`), and a
weights binding into a single, inspectable object. Both `LocalHFBackend` and
`OpenAIBackend` accept a composed `Adapter` directly via `add_adapter` —
dispatching on the weights binding's reality (`LocalFileBinding` for
LocalFile/PEFT, `EmbeddedBinding` for Embedded/Granite Switch) — alongside
the deprecated shim classes, which remain functional for now (Epic #929,
issue #1144). An `EmbeddedBinding` adapter additionally requires `add_adapter`'s
`config=` argument (the raw io.yaml mapping) — `add_adapter` raises `ValueError`
without it, since that reality's config cannot be cheaply re-derived later; see
below.

Each weights binding models how its deployment turns an adapter on.
`LocalFileBinding` downloads and loads LoRA/aLoRA weights, so it exposes a
`prepare`/`activate`/`deactivate`/`release` lifecycle — driven automatically
by `add_adapter`, so a caller need not call `prepare()` itself:

```python
# Requires: mellea[hf]
from mellea.backends.adapters import Adapter, Identity, LocalFileBinding, get_io_contract
from mellea.backends.huggingface import LocalHFBackend

# LocalFile/PEFT reality — LocalHFBackend downloads and loads the weights.
hf_backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
hf_adapter = Adapter(
    identity=Identity(name="answerability", adapter_type="lora"),
    io_contract=get_io_contract("answerability"),
    weights=LocalFileBinding.from_catalog("answerability"),
)
hf_backend.add_adapter(hf_adapter)  # downloads and loads the weights
```

`EmbeddedBinding` has no weights to manage — the adapter is already part of
the served base model — so it exposes a single method, `apply_activation`,
that edits the outgoing request instead of a lifecycle. Its `io.yaml` config
comes from the served checkpoint's `adapter_index.json`, not from anything
you can construct by hand, so registration goes through
`register_embedded_adapter_model` (or `resolve_adapter`) rather than a bare
`add_adapter(adapter)` call with no `config=`:

```python
from mellea.backends.openai import OpenAIBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW

switch_backend = OpenAIBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name,
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    load_embedded_adapters=False,
)
# Discovers "answerability" from the model's Hugging Face repo and composes
# an Adapter (Identity + IOContract + EmbeddedBinding) for it, including the
# io.yaml config a bare Adapter(weights=EmbeddedBinding.from_base_model(...))
# construction has no way to supply.
switch_backend.register_embedded_adapter_model(
    IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name, intrinsic_name="answerability"
)
```

Weights-binding support by backend today:

| Backend | `LocalFileBinding` (LocalFile/PEFT) | `EmbeddedBinding` (Embedded/Granite Switch) | `ServerMediatedBinding` |
| --- | --- | --- | --- |
| `LocalHFBackend` | ✅ shipping — `add_adapter` accepts a composed `Adapter` or a bare `LocalFileBinding` directly | ✅ shipping — `load_embedded_adapters=True`, or `add_adapter(adapter, config=...)`/`register_embedded_adapter_model` with a composed `Adapter` | — |
| `OpenAIBackend` | — | ✅ shipping — `load_embedded_adapters=True`, or `add_adapter(adapter, config=...)`/`register_embedded_adapter_model` with a composed `Adapter` | — |

`ServerMediatedBinding` has no backend implementation yet — see discussion #1486.
Discovering *multiple* embedded adapters from a Granite Switch checkpoint or
Hub repo (rather than one already-known name) still goes through
`register_embedded_adapter_model`, which builds the composed `Adapter`
instances for you.

---

## Guardian adapter functions

Safety and factuality checks use a separate set of Guardian-specific adapter functions:
`guardian_check()`, `policy_guardrails()`, `factuality_detection()`, and
`factuality_correction()`. These are documented in the
[Safety Guardrails](../how-to/safety-guardrails) how-to guide.

**See also:**
[Adding a custom adapter function in 20 lines](../tutorials/07-custom-adapter-function.md) |
[Handling a breaking adapter schema change](../tutorials/08-adapter-schema-migrations.md) |
[Adapter function metrics](../observability/metrics.md#adapter-function-metrics)
