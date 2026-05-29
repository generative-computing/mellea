---
title: "Prefix Caching and KV Blocks"
description: "Reuse KV cache state across calls to eliminate redundant prefill work on LocalHFBackend."
# diataxis: how-to
---

Prefix caching lets `LocalHFBackend` store the key-value (KV) attention states from
a forward pass and reuse them in later calls, skipping the prefill computation for
content that hasn't changed. This is useful when many calls share a large common
prefix — a system prompt, a long document, or a fixed instruction header.

**Prerequisite:** This feature is specific to `LocalHFBackend`. Server-side backends
(Ollama, OpenAI, vLLM) manage their own KV caching internally.

## Enable caching on the backend

Pass a `SimpleLRUCache` to `LocalHFBackend` at construction time:

```python
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.cache import SimpleLRUCache

backend = LocalHFBackend(
    model_id="ibm-granite/granite-3.3-2b-instruct",
    cache=SimpleLRUCache(capacity=5),
)
```

`capacity` is the maximum number of cached KV blocks held in GPU memory at once.
When the cache is full, the least recently used block is evicted and its GPU memory
freed automatically.

To disable caching entirely (useful for benchmarking):

```python
backend = LocalHFBackend(
    model_id="ibm-granite/granite-3.3-2b-instruct",
    use_caches=False,
)
```

## Mark a CBlock for caching

Caching is opt-in at the content level. Set `cache=True` on a `CBlock` to tell the
backend to prefill that block and store its KV state:

```python
from mellea.core.base import CBlock

system_doc = CBlock("You are a medical triage assistant. Always respond in structured JSON.", cache=True)
```

On the first call that includes this `CBlock`, the backend runs a forward pass and
stores the resulting `DynamicCache`. On subsequent calls containing the same block,
the cached states are retrieved and merged with the non-cached suffix — no
redundant prefill.

## How KV smashing works

When a prompt contains a mix of cached and uncached blocks, Mellea:

1. Tokenises each block independently.
2. Runs forward passes on uncached blocks.
3. Retrieves stored `DynamicCache` for cached blocks.
4. **Smashes** (concatenates) all KV caches along the time axis using
   `merge_dynamic_caches()`.
5. Passes the merged cache plus the combined input IDs to the generation step.

The result is identical to a single full-context forward pass, with the prefill
cost of cached blocks paid only once.

## Practical example

A pipeline that applies the same long grounding document to many different queries:

```python
import mellea
from mellea.core.base import CBlock
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.cache import SimpleLRUCache
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(
    model_id="ibm-granite/granite-3.3-2b-instruct",
    cache=SimpleLRUCache(capacity=3),
)
m = mellea.MelleaSession(backend=backend, ctx=ChatContext())

# This large document block will be prefilled and cached on first use.
reference = CBlock(open("large_reference_doc.txt").read(), cache=True)

queries = [
    "What are the contraindications listed?",
    "Summarise the dosage table.",
    "List any drug interactions mentioned.",
]

for query in queries:
    result = m.instruct(
        "Using the reference document, answer: {{query}}",
        user_variables={"query": query},
        grounding_context={"reference": reference},
    )
    print(str(result))
    # Output will vary — LLM responses depend on model and temperature.
```

The `reference` block is prefilled once. Each subsequent query pays only for its
own suffix tokens.

## Cache capacity and memory

Each cached block occupies GPU memory proportional to the block's token count and
the model's number of layers and attention heads. Choose `capacity` conservatively:

- **1–3** for large documents or long system prompts on a single GPU.
- **5–10** for short, frequently reused blocks with ample VRAM.

The `on_evict` callback (used internally by `LocalHFBackend`) frees GPU tensors
when a block is evicted, so the cache does not leak memory.

## Disable for benchmarking

To measure true generation time without cache benefits:

```python
backend.use_caches = False
```

Or pass `use_caches=False` at construction. The session behaviour is otherwise
identical — disabling caching only affects whether prefill states are stored and
reused.

**See also:** [HuggingFace Transformers](../integrations/huggingface) |
[Intrinsics](./intrinsics) |
[LoRA and aLoRA Adapters](./lora-and-alora-adapters)
