# Granite Switch Examples

This directory contains examples for running Mellea intrinsics through an
OpenAI-compatible backend using Granite Switch models.

## What is Granite Switch?

Granite Switch models ship with LoRA and aLoRA adapters pre-baked into the model
weights. Instead of loading adapters at runtime (as `LocalHFBackend` does), these
embedded adapters are activated via control tokens injected by the model's chat
template. Only the I/O transformation configs are downloaded — no adapter weights
are transferred.

## Prerequisites

1. A Granite Switch model hosted via [vLLM](https://docs.vllm.ai/):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <granite-switch-model-id> \
    --dtype bfloat16 \
    --enable-prefix-caching
```

2. `pip install mellea`

## Available adapters

Not all intrinsics are embedded in every Granite Switch model. Check the model's
`adapter_index.json` for the list of available adapters. The current model
includes: `answerability`, `citations`, `context_relevance`, `guardian-core`,
`hallucination_detection`, `query_clarification`, `query_rewrite`, and
`requirement-check`.

## Files

### answerability_openai.py

Demonstrates `rag.check_answerability()` using `OpenAIBackend` with
`load_embedded_adapters=True` — the simplest way to use intrinsics with Granite
Switch.

### hallucination_detection_openai.py

Demonstrates `rag.flag_hallucinated_content()` using `OpenAIBackend` with
`load_embedded_adapters=True`.

### manual_adapter_loading.py

Shows how to manually load embedded adapters using
`EmbeddedIntrinsicAdapter.from_hub()` and `backend.add_adapter()`. Useful when
you only need a subset of adapters or want more control over adapter
registration.

## Related

- [`../intrinsics/`](../intrinsics/) — the same intrinsics using `LocalHFBackend`
- [Intrinsics documentation](../../docs/docs/advanced/intrinsics.md)
