---
title: "LoRA and aLoRA adapters"
description: "Train lightweight adapters on your own labeled data and use them as requirement validators in Mellea programs."
# diataxis: how-to
---

Off-the-shelf language models sometimes fail on domain-specific tasks — particularly
requirement validation over proprietary terminology or specialized classification
schemes not well-represented in general training data. Mellea lets you train a
[LoRA](https://arxiv.org/abs/2106.09685) or
[aLoRA](https://github.com/IBM/activated-lora) adapter on your own labeled dataset
and use it as a requirement validator in any Mellea program.

**Prerequisites:** `pip install "mellea[cli,hf]"`. Training requires a GPU or
Apple Silicon Mac with sufficient VRAM for the chosen base model. Uploading requires a
Hugging Face account.

> **Backend note:** Custom-trained adapters can only be loaded directly into
> `LocalHFBackend`. Ollama can use a custom adapter bundled into a model with a
> Modelfile `ADAPTER` line, but Mellea does not discover custom Ollama adapter
> functions from `adapter_models` alone. Register a composed adapter with its
> `io.yaml` explicitly, then map its name to the bundled model tag. See
> [Adapter functions](./intrinsics.md) for the supported catalogue-adapter
> workflow.
>
> Granite Switch models ship with pre-trained adapter functions embedded in the
> model weights. Use them through `OpenAIBackend` with a served checkpoint, or
> through `LocalHFBackend` with a local checkpoint and
> `load_embedded_adapters=True`. See [Adapter functions](./intrinsics.md) for details.

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

If you intend to use the adapter as a Mellea adapter function (so that it can be loaded by
model ID rather than local path), pass `--intrinsic` and provide an `io.yaml` file:

```bash
m alora upload ./checkpoints/my_adapter \
  --name your-org/my-adapter \
  --intrinsic \
  --io-yaml ./io.yaml
```

## Use the adapter in Mellea

Load the trained adapter into a `LocalHFBackend` by composing an `Adapter`
directly — `LocalFileBinding` accepts an arbitrary `repo_id`, so a locally
trained, non-catalog adapter does not need a dedicated shim class.

```python
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.adapters import Adapter, Identity, LocalFileBinding, get_io_contract
from mellea.backends.adapters.catalog import AdapterType
from mellea.stdlib.context import ChatContext
from mellea import MelleaSession
from mellea.stdlib.requirements import ALoraRequirement

backend = LocalHFBackend(model_id="ibm-granite/granite-3.2-8b-instruct")

adapter = Adapter(
    identity=Identity(name="custom-failure-check", adapter_type="alora"),
    # get_io_contract falls back to a permissive dict contract for names
    # outside the built-in catalog, so this works for a custom adapter too.
    io_contract=get_io_contract("custom-failure-check"),
    weights=LocalFileBinding(
        name="custom-failure-check",
        adapter_type=AdapterType.ALORA,
        repo_id="your-org/my-adapter",  # HF repo ID or local checkpoint path
        revision="main",  # a custom adapter has no catalog entry to resolve this from
    ),
)
backend.add_adapter(adapter)  # downloads and loads the weights

m = MelleaSession(backend, ctx=ChatContext())

failure_check = ALoraRequirement(
    "The failure mode must not be 'no_failure'.",
    intrinsic_name="custom-failure-check",
    # Required for a name outside the intrinsics catalog — see Intrinsic.
    adapter_types=(AdapterType.ALORA,),
)
result = m.instruct(
    "Write a triage summary based on this technician note: {{note}}",
    user_variables={"note": "High vibration at 3100 RPM, connecting rod suspected."},
    requirements=[failure_check],
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

`ALoraRequirement` routes validation through the adapter with the matching
`intrinsic_name`. Call `backend.add_adapter(adapter)` before the requirement
runs — it registers the custom name so `ALoraRequirement` can resolve it. The
adapter runs at the `check_requirement` prompt position. Its `io.yaml` must
transform the output into the `{"requirement_check": {"score": <float>}}`
response schema; label-only adapter output is not compatible with
`ALoraRequirement`.

## How automatic routing works

When an adapter is loaded via `backend.add_adapter()`, Mellea automatically routes
`req()` validation calls through it rather than falling back to LLM-as-a-judge. The
rule is: use the most specific available method. In practice this means the aLoRA
adapter is preferred whenever one is loaded, with three exceptions:

1. `backend.default_to_constraint_checking_alora` is set to `False` — the adapter
   is loaded but routing is suppressed for the entire backend instance.
2. The requirement uses the `LLMaJRequirement` subtype explicitly — the caller is
   asking for LLM-as-a-judge regardless of what adapters are loaded.
3. The adapter is unavailable — Mellea falls back to LLM-as-a-judge
   automatically. This covers the adapter not being registered at all, and
   (see [aLoRA activation is not guaranteed by
   loading](#alora-activation-is-not-guaranteed-by-loading)) an aLoRA that is
   registered but proven unable to activate for this prompt: generation for
   it is skipped entirely, so it never reaches the case below. This is the
   *only* fallback case: if the adapter genuinely runs and its output fails
   schema validation, `validate()` does not fall back. Instead it surfaces
   the schema error on `ValidationResult.error` and fails the check closed
   (`bool(result)` is `False`), so callers can tell an unparsable adapter
   response apart from an ordinary "requirement not met".

If you want to force the adapter path even when using `generate_from_context`
directly (bypassing the normal `validate()` call), use `ALoraRequirement` from
`mellea.stdlib.requirements` — this bypasses `default_to_constraint_checking_alora`,
but still requires a matching adapter to actually be registered. If none is found,
Mellea logs a warning and falls back to regular generation rather than erroring.

### What the adapter sees

The `requirement-check` adapter judges the last assistant turn of the conversation it is
given, so the routing above only produces useful verdicts if that conversation actually
reaches it. Mellea builds the adapter's message list from the validation context's
`view_for_generation()`, and validation runs over the post-generation context — the same
conversation the model generated into, with the generated output last.

That only works on a context that renders history, which the default session does not
provide: `start_session()` returns a session backed by `SimpleContext`, which retains
`last_output()` but whose `view_for_generation()` is always empty by design. An
adapter-backed requirement has no conversation to judge there, so an explicit
`ALoraRequirement` raises `ValueError` and a plain `Requirement` logs a warning and falls
back to LLM-as-a-judge. Ask for a chat context explicitly:

```python
import mellea

# Adapter-backed requirements need a context that renders the assistant turn.
m = mellea.start_session(context_type="chat")
```

See [What the validator sees](../concepts/requirements-system.md#what-the-validator-sees).

### aLoRA activation is not guaranteed by loading

An aLoRA only takes effect from the point its declared invocation token sequence appears
in the assembled prompt. If that sequence is absent, PEFT switches the adapter off
silently: generation still runs, still returns a well-formed score, and
`list_adapters()`/`active_adapters()` still show the adapter as loaded, but the base
model produced the output, not the adapter. Mellea checks for this on every generation
call for an aLoRA registered as a composed `Adapter` (the default `resolve_adapter()`
path, and what every high-level adapter-function wrapper uses), before any model call
runs, and raises `AloraActivationError` (logging a warning the first time, naming the
capability, the base model, and the decoded invocation sequence) rather than letting a
fabricated result reach a caller. What happens next depends on how the adapter was
reached:

- **Automatic `Requirement` routing** (a plain `Requirement`, or an explicit
  `ALoraRequirement`) catches the error and falls back to LLM-as-a-judge, logging a
  warning -- exactly the same treatment as "adapter not registered" already gets. Only
  an `ALoraRequirement` over a context with nothing to judge still raises; this is not
  that case.
- **A direct adapter-function call** (e.g. `core.check_certainty`) has no fallback
  mechanism below it and lets `AloraActivationError` propagate, so the caller sees a
  clear error instead of a silently wrong score.

This check only runs for the LocalFile/PEFT reality (e.g. `LocalHFBackend`);
server-mediated and embedded adapter deployments activate outside Mellea and are not
covered, and the deprecated `IntrinsicAdapter` shim's weights load too late in the call
for the check to see them (Epic #929, issue #1144 tracks removing the shim entirely).

## Disable adapter validation

To run without adapter validation (for benchmarking or debugging):

```python
backend.default_to_constraint_checking_alora = False
```

Set it back to `True` to re-enable. This flag is per-backend instance and does not
affect other sessions.

**See also:** [Adapter functions](./intrinsics.md) |
[The Requirements System](../concepts/requirements-system.md) |
[Write Custom Verifiers](../how-to/write-custom-verifiers.md) |
[CLI Reference](../reference/cli.md)
