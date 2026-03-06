---
title: "HuggingFace and vLLM"
description: "Run Mellea on local GPU hardware with LocalHFBackend (HuggingFace Transformers) or LocalVLLMBackend (vLLM)."
# diataxis: how-to
---

# HuggingFace and vLLM

Mellea provides two local inference backends for running models directly on your
own hardware: `LocalHFBackend` (HuggingFace Transformers) and `LocalVLLMBackend`
(vLLM). Both download model weights on first use and run inference locally — no
cloud credentials required.

| | `LocalHFBackend` | `LocalVLLMBackend` |
|---|---|---|
| Install extra | `mellea[hf]` | `mellea[vllm]` |
| Platform | macOS, Linux, Windows | Linux only |
| Device | cuda > mps > cpu (auto) | cuda required |
| Best for | Experimental features (aLoRA, constrained decoding) | High-throughput batched inference |
| aLoRA support | Yes | Planned |

> **Tip:** For everyday local inference without experimental features, use
> [Ollama](./ollama.md) — it is simpler to set up and well suited for development.

---

## LocalHFBackend

`LocalHFBackend` uses [HuggingFace Transformers](https://huggingface.co/docs/transformers)
for inference. It is designed for experimental Mellea features — aLoRA adapters,
constrained decoding, and span-based context — that are not yet available on
server-based backends.

**Install:**

```bash
pip install 'mellea[hf]'
```

### Basic usage

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

### Device selection

The backend selects the device automatically: CUDA GPU if available, then Apple
Silicon MPS, then CPU. To override device selection, use `custom_config`:

```python
from mellea.backends.huggingface import LocalHFBackend, TransformersTorchConfig

m_backend = LocalHFBackend(
    "ibm-granite/granite-3.3-8b-instruct",
    custom_config=TransformersTorchConfig(device="cpu"),
)
```

### KV cache

`LocalHFBackend` caches KV blocks across calls by default (`use_caches=True`). This
speeds up repeated calls that share a common prefix. Disable it for debugging:

```python
m_backend = LocalHFBackend(model_ids.IBM_GRANITE_4_HYBRID_MICRO, use_caches=False)
```

### aLoRA adapters

`LocalHFBackend` supports [Activated LoRA (aLoRA)](../advanced/lora-and-alora-adapters.md)
adapters — lightweight domain-specific requirement validators that run on local GPU
hardware. See the aLoRA guide for training and usage.

---

## LocalVLLMBackend

`LocalVLLMBackend` uses [vLLM](https://vllm.ai/) for higher-throughput local inference.
It is a good choice when you are running many requests in parallel (e.g., batch
evaluation). vLLM takes longer to initialise than `LocalHFBackend` but sustains higher
throughput once warm.

**Install (Linux only):**

```bash
pip install 'mellea[vllm]'
```

> **Platform note:** vLLM is not supported on macOS. Use `LocalHFBackend` or Ollama
> on Apple Silicon.

### Getting started with vLLM

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

### High-throughput batched inference

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

---

## Troubleshooting

### `pip install mellea[hf]` fails on Intel macOS

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

### vLLM: output truncated at ~16 tokens

vLLM defaults to approximately 16 tokens. Set `ModelOption.MAX_NEW_TOKENS` explicitly:

```python
model_options={ModelOption.MAX_NEW_TOKENS: 512}
```

---

**Previous:** [AWS Bedrock and IBM watsonx](./bedrock-and-watsonx.md) |
**Next:** [MCP Integration](./mcp.md)

**See also:** [Backends and Configuration](../guide/backends-and-configuration.md) |
[LoRA and aLoRA Adapters](../advanced/lora-and-alora-adapters.md)
