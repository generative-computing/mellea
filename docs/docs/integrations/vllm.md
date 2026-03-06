---
title: "vLLM"
description: "Run Mellea with high-throughput local inference using LocalVLLMBackend and vLLM."
# diataxis: how-to
---

# vLLM

`LocalVLLMBackend` uses [vLLM](https://vllm.ai/) for higher-throughput local inference.
It is a good choice when you are running many requests in parallel — for example, batch
evaluation or load testing. vLLM takes longer to initialise than `LocalHFBackend` but
sustains higher throughput once warm.

**Prerequisites:** `pip install 'mellea[vllm]'`, Linux, CUDA GPU.

> **Platform note:** vLLM is not supported on macOS. Use
> [`LocalHFBackend`](./huggingface) or [Ollama](./ollama) on Apple Silicon.

## Install

```bash
pip install 'mellea[vllm]'
```

## Basic usage

```python
from mellea import MelleaSession
from mellea.backends import ModelOption, model_ids
from mellea.backends.vllm import LocalVLLMBackend

m = MelleaSession(
    LocalVLLMBackend(
        model_ids.IBM_GRANITE_4_HYBRID_MICRO,
        model_options={ModelOption.MAX_NEW_TOKENS: 256},
    )
)

result = m.instruct("Explain the difference between precision and recall.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

> **Always set `MAX_NEW_TOKENS` explicitly.** vLLM defaults to approximately 16 tokens.
> For structured output or longer responses, set `ModelOption.MAX_NEW_TOKENS` to
> 200–1000+ tokens.

## High-throughput batched inference

vLLM processes requests in continuous batches. For batch evaluation, send requests
concurrently rather than sequentially to take advantage of the batching:

```python
import asyncio
from mellea import MelleaSession
from mellea.backends import ModelOption, model_ids
from mellea.backends.vllm import LocalVLLMBackend

backend = LocalVLLMBackend(
    model_ids.IBM_GRANITE_4_HYBRID_MICRO,
    model_options={ModelOption.MAX_NEW_TOKENS: 512},
)

async def run_batch(prompts: list[str]) -> list[str]:
    m = MelleaSession(backend)
    tasks = [m.ainstruct(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return [str(r) for r in results]
```

## Vision support

Vision support for `LocalVLLMBackend` is model-dependent. Pass a PIL image or an
[`ImageBlock`](../guide/glossary#imageblock) via `images=[...]` when using a
vision-capable model. See [Use Images and Vision Models](../how-to/use-images-and-vision).

## Troubleshooting

### Output truncated at ~16 tokens

vLLM defaults to approximately 16 tokens. Set [`ModelOption`](../guide/glossary#modeloption)
`MAX_NEW_TOKENS` explicitly:

```python
model_options={ModelOption.MAX_NEW_TOKENS: 512}
```

---


**See also:** [Backends and Configuration](../guide/backends-and-configuration) |
[LoRA and aLoRA Adapters](../advanced/lora-and-alora-adapters)
