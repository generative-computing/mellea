# aLoRA Examples

This directory contains examples demonstrating how to tune and use your own (Activated)
Low-Rank Adapters. For the full CLI reference and a conceptual overview, see
[LoRA and aLoRA adapters](../../docs/advanced/lora-and-alora-adapters.md),
[`m alora` CLI reference](../../docs/reference/cli.md), and the
[custom adapter function tutorial](../../docs/tutorials/07-custom-adapter-function.md).
For the built-in catalog adapters, see [`intrinsics/`](../intrinsics/README.md).

## Examples

| File | Description |
| :--- | :--- |
| [`101_example.py`](./101_example.py) | Comparing aLoRA-backed requirement validation against LLM-as-a-judge, using the catalog-native `requirement-check` adapter (not a custom one — see `stembolts_intrinsic.py` for loading your own). |
| [`102_example.py`](./102_example.py) | Interactive loop that exercises the custom `stembolts` adapter — skip-marked because it blocks on stdin in an infinite loop. |
| [`stembolts_intrinsic.py`](./stembolts_intrinsic.py) | Helper module for loading and calling the fully custom `stembolts` adapter (no standalone entry point — `102_example.py` consumes it). |
| [`make_training_data.py`](./make_training_data.py) | Utility for preparing training datasets for adapter training. |
| [`example_readme_generator.py`](./example_readme_generator.py) | Programmatic README generation for adapters. |
| [`stembolt_failure_dataset.jsonl`](./stembolt_failure_dataset.jsonl) | Training data in the `{item, label}` format that `m alora train` expects. |
| [`stembolt_customer_data.jsonl`](./stembolt_customer_data.jsonl) | Raw data used as input to `make_training_data.py`. |
| [`io.yaml`](./io.yaml) | Output contract for the `stembolts` adapter, passed to `m alora upload --io-yaml`. |
| [`hints.txt`](./hints.txt) | Domain hints passed to `m alora add-readme --hints`. |

## Training Models

First, we need to clone Mellea and put ourselves into this directory:

```bash
git clone https://github.com/generative-computing/mellea;
cd mellea;
uv venv .venv;
source .venv/bin/activate;
uv pip install -e ".[all]";
pushd docs/examples/aLora/;
```

Now let's train a model:

```bash
m alora train \
    --basemodel ibm-granite/granite-4.1-3b \
    --outfile stembolts_model \
    --adapter alora \
    stembolt_failure_dataset.jsonl
```

> [!NOTE]
> You will need hardware capable of training models.
> For local training, our minimum recommendation is an M1 MAX with 64GB unified memory. This will allow you to train small language model adapters.
> Alternatively, you can train small language models on relatively cheap spot instances at many popular cloud providers.

## Upload Models

If model training succeeds, you will need to upload your model as an adapter function.
Uploading requires a Hugging Face account; run `huggingface-cli login` first, or
`m alora upload` and `m alora add-readme` fail with "Hugging Face token not found".

```bash
# WARNING: running this command will upload your model weights to huggingface.co !!!
# The model will be private.
# replace $HF_USERNAME with your Hugging Face username.
m alora upload \
    --intrinsic \
    --name "$HF_USERNAME/stembolts" \
    --io-yaml io.yaml \
    stembolts_model
```

You can also train and upload the same adapter for multiple model families:

```bash
# CHANGE $HF_USERNAME to your username, or set envvar.
m alora train \
    --basemodel ibm-granite/granite-3.3-2b-instruct \
    --outfile stembolts_model_3.3_2b \
    --adapter alora \
    stembolt_failure_dataset.jsonl &&
m alora upload \
    --intrinsic \
    --name "$HF_USERNAME/stembolts" \
    --io-yaml io.yaml \
    stembolts_model_3.3_2b
```

## Generate a README

After uploading your adapter, you can auto-generate a README for the Hugging Face model repository using `m alora add-readme`. This command uses Mellea to analyze your training dataset and produce documentation with a description, data examples, and integration code. It needs a running model backend (the default session backend, so `ollama serve`):

```bash
m alora add-readme \
    --name $HF_USERNAME/stembolts \
    --io-yaml io.yaml \
    --basemodel granite-4.1-3b \
    stembolt_failure_dataset.jsonl
```

You can provide a `--hints` file with additional domain context to improve the generated descriptions:

```bash
m alora add-readme \
    --name $HF_USERNAME/stembolts \
    --io-yaml io.yaml \
    --basemodel granite-4.1-3b \
    --hints hints.txt \
    stembolt_failure_dataset.jsonl
```

The generator will display the README and ask for confirmation before uploading it to your Hugging Face repo. You can also call the generator programmatically from Python — see `example_readme_generator.py` for an example.

## Using Adapter Functions

Compose the adapter from an `Identity`, an output contract, and a `LocalFileBinding`,
then register it on a `LocalHFBackend` with `backend.add_adapter()`. The code is in
[LoRA and aLoRA adapters](../../docs/advanced/lora-and-alora-adapters.md), and
`stembolts_intrinsic.py` is a complete, runnable version with the `stembolts` adapter;
replace its `repo_id` (`nfulton/stembolts`) with `$HF_USERNAME/stembolts` to use your
own upload. A custom, non-catalog adapter has no catalog entry to resolve `revision`
from, so pin it explicitly (`revision="main"`).
