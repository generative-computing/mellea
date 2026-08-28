# Granite Switch Examples

This directory contains examples for running Mellea adapter functions with
Granite Switch models, either locally through `LocalHFBackend` or through an
OpenAI-compatible vLLM deployment.

## What is Granite Switch?

Granite Switch models ship with LoRA and aLoRA adapters pre-baked into the model
weights. Unlike runtime LoRA/aLoRA adapters on a standard `LocalHFBackend`,
these embedded adapters are activated via control tokens injected by the model's
chat template. Only the I/O transformation configs are downloaded — no adapter
weights are transferred.

## Prerequisites

### Local Hugging Face inference

Run the local example on a GPU or Apple Silicon Mac:

```bash
uv sync --extra hf
```

### OpenAI-compatible inference

1. Host a Granite Switch model with [vLLM](https://docs.vllm.ai/).
2. Install the `switch` extra to download embedded adapter metadata:

```bash
uv sync --extra switch
```

## Available adapters

Not all adapter functions are embedded in every Granite Switch model. Check the
model's `adapter_index.json` for a definitive list. For Granite Switch models
pre-built by IBM, Mellea includes a list of models in `model_id`.

## Files

### answerability_local_hf.py

Demonstrates `rag.check_answerability()` against a local Granite Switch
checkpoint using `LocalHFBackend(load_embedded_adapters=True)`.

### answerability_openai.py

Demonstrates `rag.check_answerability()` using `OpenAIBackend` with
`load_embedded_adapters=True` — the simplest way to use adapter functions with Granite
Switch.

### hallucination_detection_openai.py

Demonstrates `rag.flag_hallucinated_content()` using `OpenAIBackend` with
`load_embedded_adapters=True`.

### manual_adapter_loading.py

Shows how to manually load embedded adapters using
`EmbeddedIntrinsicAdapter.from_hub()` and `backend.add_adapter()`. Useful when
you only need a subset of adapters or want more control over adapter
registration.

## Architecture

![Granite Libraries Software Stack Architecture in Mellea](../../docs/images/granite-libraries-mellea-architecture.png)

## Related

- [`../intrinsics/`](../intrinsics/) — runtime LoRA/aLoRA adapter functions
- [Adapter Functions Documentation](../../docs/docs/advanced/intrinsics.md)
- [Official Granite Switch Documentation](https://github.com/generative-computing/granite-switch)
