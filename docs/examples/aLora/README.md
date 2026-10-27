# aLoRA Examples

Examples demonstrating how to train, upload, and use your own (Activated) Low-Rank
Adapters as Mellea adapter functions, using the fictional "stembolt" repair-note
dataset. The commands below are the workflow that goes with the scripts in the
table. For the full CLI reference and a conceptual overview, see
[LoRA and aLoRA adapters](../../docs/advanced/lora-and-alora-adapters.md),
[`m alora` CLI reference](../../docs/reference/cli.md), and the
[custom adapter function tutorial](../../docs/tutorials/07-custom-adapter-function.md).
For the built-in catalog adapters, see [`intrinsics/`](../intrinsics/README.md).

## Examples

| File | Description |
| :--- | :--- |
| [`101_example.py`](./101_example.py) | Comparing `ALoraRequirement` against `LLMaJRequirement` validation, with timing, using the catalog `requirement-check` adapter. |
| [`102_example.py`](./102_example.py) | Interactive loop that calls the custom `stembolts` adapter. Skipped in automated runs because it blocks on stdin. |
| [`stembolts_intrinsic.py`](./stembolts_intrinsic.py) | Helper module (no entry point) that composes the custom `stembolts` `Adapter`, defines `StemboltIntrinsic`, and calls it through `mfuncs.act()`. Consumed by `102_example.py`. |
| [`make_training_data.py`](./make_training_data.py) | Converting raw JSONL (`item`, `defective_part`, `diag_likelihood`) into the `{item, label}` training format with schema validation. |
| [`example_readme_generator.py`](./example_readme_generator.py) | Generating a Hugging Face README for an adapter programmatically, the Python equivalent of `m alora add-readme`. |
| [`stembolt_failure_dataset.jsonl`](./stembolt_failure_dataset.jsonl) | Training data in the `{item, label}` format that `m alora train` expects. |
| [`stembolt_customer_data.jsonl`](./stembolt_customer_data.jsonl) | Raw data used as input to `make_training_data.py`. |
| [`io.yaml`](./io.yaml) | Output contract for the `stembolts` adapter, passed to `m alora upload --io-yaml`. |
| [`hints.txt`](./hints.txt) | Domain hints passed to `m alora add-readme --hints`. |

## Workflow

### Prerequisites

Clone Mellea, install it with the CLI and Hugging Face extras, and move into this
directory:

```bash
git clone https://github.com/generative-computing/mellea
cd mellea
uv sync --extra cli --extra hf
cd docs/examples/aLora
```

You need hardware capable of training models. For local training, our minimum
recommendation is an M1 Max with 64GB unified memory, which is enough for small
language model adapters. Alternatively, train small language models on relatively
cheap spot instances at popular cloud providers. Uploading requires a Hugging Face
account; authenticate first with `huggingface-cli login`, or `m alora upload` and
`m alora add-readme` fail with "Hugging Face token not found".

### Train

```bash
uv run m alora train \
    --basemodel ibm-granite/granite-4.1-3b \
    --outfile stembolts_model \
    --adapter alora \
    stembolt_failure_dataset.jsonl
```

### Upload

If training succeeds, upload the model as an adapter function. This uploads your
model weights to huggingface.co (the repository is private). Replace
`$HF_USERNAME` with your Hugging Face username.

```bash
uv run m alora upload \
    --intrinsic \
    --name "$HF_USERNAME/stembolts" \
    --io-yaml io.yaml \
    stembolts_model
```

You can train and upload the same adapter for multiple model families:

```bash
uv run m alora train \
    --basemodel ibm-granite/granite-3.3-2b-instruct \
    --outfile stembolts_model_3.3_2b \
    --adapter alora \
    stembolt_failure_dataset.jsonl &&
uv run m alora upload \
    --intrinsic \
    --name "$HF_USERNAME/stembolts" \
    --io-yaml io.yaml \
    stembolts_model_3.3_2b
```

### Generate a README

`m alora add-readme` uses Mellea to analyze your training dataset and produce a
Hugging Face model README with a description, data examples, and integration code.
It displays the README and asks for confirmation before uploading it. It needs a
running model backend (the default session backend, so `ollama serve`).

```bash
uv run m alora add-readme \
    --name $HF_USERNAME/stembolts \
    --io-yaml io.yaml \
    --basemodel granite-4.1-3b \
    stembolt_failure_dataset.jsonl
```

Pass `--hints hints.txt` to give the generator additional domain context:

```bash
uv run m alora add-readme \
    --name $HF_USERNAME/stembolts \
    --io-yaml io.yaml \
    --basemodel granite-4.1-3b \
    --hints hints.txt \
    stembolt_failure_dataset.jsonl
```

### Use your adapter

Your adapter is composed from an `Identity`, an output contract, and a
`LocalFileBinding`, then registered on a `LocalHFBackend` with
`backend.add_adapter()`. `stembolts_intrinsic.py` is a complete, runnable version
of this with the `stembolts` adapter; replace its `repo_id` (`nfulton/stembolts`)
with `$HF_USERNAME/stembolts` to use your own upload. A custom, non-catalog
adapter has no catalog entry to resolve `revision` from, so pin it explicitly
(`revision="main"`).
