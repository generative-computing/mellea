---
title: "LoRA and aLoRA adapters"
description: "Train lightweight adapters on your own labeled data and use them as requirement validators in Mellea programs."
# diataxis: how-to
---

# LoRA and aLoRA adapters

Off-the-shelf language models sometimes fail on domain-specific tasks — particularly
requirement validation over proprietary terminology or specialized classification
schemes not well-represented in general training data. Mellea lets you train a
[LoRA](https://arxiv.org/abs/2106.09685) or
[aLoRA](https://github.com/IBM/activated-lora) adapter on your own labeled dataset
and use it as a requirement validator in any Mellea program.

**Prerequisites:** `pip install mellea`, `m` CLI available. Training requires a GPU or
Apple Silicon Mac with sufficient VRAM for the chosen base model. Uploading requires a
Hugging Face account.

> **Backend note:** Trained adapters can only be loaded into `LocalHFBackend`. They do
> not work with Ollama, OpenAI, or other remote backends.

## LoRA vs aLoRA

Both adapter types fine-tune a base model on your data. The difference is inference cost:

| | LoRA | aLoRA |
| --- | --- | --- |
| Inference overhead | Processes full context each call | Activated at a single token — minimal overhead |
| Best for | General fine-tuning | Fast inner-loop checks, requirement validation |
| Training time | Similar | Similar |

For requirement validation in Mellea (short binary checks inside a generation loop),
aLoRA is the better choice. Use `--adapter lora` if you need a more general fine-tune
and can absorb the inference cost.

## Data format

Training data is a `.jsonl` file with one JSON object per line. Each object must have:

- `item` — the input text to classify
- `label` — the string classification label

```json
{"item": "Observed black soot on intake. Seal seems compromised under thermal load.", "label": "piston_rings"}
{"item": "Rotor misalignment caused torsion on connecting rod. High vibration at 3100 RPM.", "label": "connecting_rod"}
{"item": "Combustion misfire traced to a cracked mini-carburetor flange.", "label": "mini_carburetor"}
{"item": "Stembolt makes a whistling sound and does not complete the sealing process.", "label": "no_failure"}
```

Labels can be any strings. The adapter learns to predict the label from the item text.

## Train an adapter

```bash
m alora train data.jsonl \
  --basemodel ibm-granite/granite-3.2-8b-instruct \
  --outfile ./checkpoints/my_adapter \
  --adapter alora \
  --epochs 6 \
  --learning-rate 6e-6 \
  --batch-size 2 \
  --max-length 1024 \
  --grad-accum 4
```

The trained adapter weights are saved to `./checkpoints/my_adapter/`.

### Parameters

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `datafile` | `str` | required | Path to `.jsonl` training file |
| `--basemodel` | `str` | required | Hugging Face model ID or local path |
| `--outfile` | `str` | required | Directory to save adapter weights |
| `--adapter` | `str` | `alora` | Adapter type: `alora` or `lora` |
| `--device` | `str` | `auto` | Device: `auto`, `cpu`, `cuda`, or `mps` |
| `--epochs` | `int` | `6` | Number of training epochs |
| `--learning-rate` | `float` | `6e-6` | Learning rate |
| `--batch-size` | `int` | `2` | Per-device batch size |
| `--max-length` | `int` | `1024` | Max tokenized sequence length |
| `--grad-accum` | `int` | `4` | Gradient accumulation steps |
| `--promptfile` | `str` | None | JSON file overriding the invocation prompt |

The default invocation prompt is `<|start_of_role|>check_requirement<|end_of_role|>`.
Provide `--promptfile` only if your adapter needs a different prompt format. The file
must contain `{"invocation_prompt": "..."}`.

## Upload to Hugging Face

```bash
huggingface-cli login  # one-time setup

m alora upload ./checkpoints/my_adapter \
  --name your-org/my-adapter
```

This creates the Hugging Face repository if it does not exist and uploads the adapter
weights. Requires `HF_TOKEN` set or a prior `huggingface-cli login`.

> **Warning:** Before uploading to a public repository, review whether your training
> data includes proprietary, confidential, or personal information. Language models can
> memorize details from small domain-specific datasets.

If you intend to use the adapter as a Mellea intrinsic (so that it can be loaded by
model ID rather than local path), pass `--intrinsic` and provide an `io.yaml` file:

```bash
m alora upload ./checkpoints/my_adapter \
  --name your-org/my-adapter \
  --intrinsic \
  --io-yaml ./io.yaml
```

## Use the adapter in Mellea

Load the trained adapter into a `LocalHFBackend` using `CustomIntrinsicAdapter`:

```python
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.adapters.adapter import CustomIntrinsicAdapter
from mellea.stdlib.context import ChatContext
from mellea import MelleaSession
from mellea.stdlib.requirements import req

backend = LocalHFBackend(model_id="ibm-granite/granite-3.2-8b-instruct")

adapter = CustomIntrinsicAdapter(
    model_id="your-org/my-adapter",       # HF repo ID or local checkpoint path
    base_model_name="granite-3.2-8b-instruct",
)
backend.add_adapter(adapter)

m = MelleaSession(backend, ctx=ChatContext())

failure_check = req("The failure mode must not be 'no_failure'.")
result = m.instruct(
    "Write a triage summary based on this technician note: {{note}}",
    user_variables={"note": "High vibration at 3100 RPM, connecting rod suspected."},
    requirements=[failure_check],
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

When `backend.add_adapter()` is called, Mellea automatically routes requirement
validation through the adapter for any `req()` calls on that session. The adapter
runs at the `check_requirement` prompt position — fast, with minimal context overhead.

## Disable adapter validation

To run without adapter validation (for benchmarking or debugging):

```python
backend.default_to_constraint_checking_alora = False
```

Set it back to `True` to re-enable. This flag is per-backend instance and does not
affect other sessions.

**See also:** [Intrinsics](./intrinsics.md) |
[The Requirements System](../concepts/requirements-system.md) |
[Write Custom Verifiers](../how-to/write-custom-verifiers.md)

---

**Previous:** [Intrinsics](./intrinsics.md) |
**Next:** [Inference-Time Scaling](./inference-time-scaling.md)
