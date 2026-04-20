---
title: "HuggingFace Transformers"
description: "Run Mellea on local hardware with LocalHFBackend and HuggingFace Transformers."
# diataxis: how-to
---

`LocalHFBackend` uses [HuggingFace Transformers](https://huggingface.co/docs/transformers)
for local inference. It is designed for experimental Mellea features — aLoRA adapters,
constrained decoding, and span-based context — that are not yet available on
server-based backends.

**Prerequisites:** `pip install 'mellea[hf]'`, Python 3.11+, local model weights.

> **Tip:** For everyday local inference without experimental features, use
> [Ollama](./ollama) — it is simpler to set up and well suited for development.

## Install

```bash
pip install 'mellea[hf]'
```

## Basic usage

```python
from mellea import MelleaSession
from mellea.backends import ModelOption, model_ids
from mellea.backends.huggingface import LocalHFBackend

m = MelleaSession(
    LocalHFBackend(
        model_ids.IBM_GRANITE_4_HYBRID_MICRO,
        model_options={ModelOption.MAX_NEW_TOKENS: 256},
    )
)

result = m.instruct("Summarize the key ideas in the theory of relativity.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

On first run, `LocalHFBackend` downloads the model weights via the Transformers
`Auto*` classes and loads them onto the best available device (cuda > mps > cpu).

## Device selection

The [`Backend`](../guide/glossary#backend) selects the device automatically: CUDA GPU
if available, then Apple Silicon MPS, then CPU. To override device selection, use
`custom_config`:

```python
from mellea.backends.huggingface import LocalHFBackend, TransformersTorchConfig

m_backend = LocalHFBackend(
    "ibm-granite/granite-3.3-8b-instruct",
    custom_config=TransformersTorchConfig(device="cpu"),
)
```

## KV cache

`LocalHFBackend` caches KV blocks across calls by default (`use_caches=True`). This
speeds up repeated calls that share a common prefix. Pass a [`SimpleLRUCache`](../guide/glossary#simplelrucache)
to control capacity, or disable caching entirely for debugging:

```python
from mellea.backends.cache import SimpleLRUCache

# Enable with explicit capacity
m_backend = LocalHFBackend(model_ids.IBM_GRANITE_4_HYBRID_MICRO, cache=SimpleLRUCache(5))

# Disable entirely
m_backend = LocalHFBackend(model_ids.IBM_GRANITE_4_HYBRID_MICRO, use_caches=False)
```

See [Prefix Caching and KV Blocks](../advanced/prefix-caching-and-kv-blocks) for full details on marking blocks for caching and how [KV smashing](../guide/glossary#kv-smashing) works.

## aLoRA adapters

`LocalHFBackend` supports [Activated LoRA (aLoRA)](../advanced/lora-and-alora-adapters)
adapters — lightweight domain-specific requirement validators that run on local GPU
hardware. See the aLoRA guide for training and usage.

> **Tip:** For intrinsics without local GPU requirements, Granite Switch models
> serve pre-embedded adapters via vLLM and the OpenAI backend. See
> [Intrinsics](../advanced/intrinsics) for details.

## Vision support

Vision support for `LocalHFBackend` is model-dependent and experimental. Pass a PIL
image or an [`ImageBlock`](../guide/glossary#imageblock) via `images=[...]` to
`instruct()` or `chat()` when using a vision-capable model. Not all models loaded via
`LocalHFBackend` support image input. See
[Use Images and Vision Models](../how-to/use-images-and-vision).

## Troubleshooting

### `pip install "mellea[hf]"` fails on Intel macOS

If you see torch/torchvision version errors on an Intel Mac, use Conda:

```bash
conda install 'torchvision>=0.22.0'
pip install mellea
```

Then run examples with `python` inside the Conda environment rather than
`uv run --with mellea`.

### Python 3.13: `error: can't find Rust compiler`

The `outlines` package (used by `mellea[hf]`) requires a Rust compiler on Python 3.13.
Either downgrade to Python 3.12 or install the
[Rust compiler](https://www.rust-lang.org/tools/install):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

---

**See also:** [Backends and Configuration](../guide/backends-and-configuration) |
[LoRA and aLoRA Adapters](../advanced/lora-and-alora-adapters)
