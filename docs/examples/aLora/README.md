# aLoRA Examples

This directory contains examples demonstrating how to tune and use your own (Adaptive) Low-Rank Adapters.

## Training Models

First, we need to clone Mellea and put ourselves into this directory:

```bash
git clone github.com/generative-computing/mellea;
cd mellea;
uv venv .venv;
source .venv/bin/activate;
uv pip install -e .[all];
pushd docs/examples/aLora/;
```

Now let's train a model:

```
m alora train \
    --basemodel ibm-granite/granite-4.0-micro \
    --outfile stembolts_model \
    --adapter alora \
    stembolt_failure_dataset.jsonl
```

>![NOTE]
> You will need hardware capable of training models. 
> For local training, our minimum recommendation is an M1 MAX with 64GB unitifed memory. This will allow you to train small language model adapters.
> Alternatively, you can train small language models on relatively cheap spot instances at many popular cloud providers.

## Upload Models

If model training succeeds, you will need to upload your model as an intrinsic:

```bash
# WARNING: running this command will upload your model weights to huggingface.co !!!
# The model will be private.
# replace $HF_USERNAME with your huggingface username.
m alora upload \
   --intrinsic \
   --name "$HF_USERNAME/stembolts" \
   --io-yaml io.yaml \
    stembolts_model
```

You can also train and upload the same adapter for multiple model families:

```
# CHANGE $HF_USERNAME to your username, or set envvar.
m alora train \
    --basemodel ibm-granite/granite-3.3-2b-instruct \
    --outfile stembolts_model_3.3_2b \
    --adapter alora \
    stembolt_failure_dataset.jsonl &&
m alora upload \
   --intrinsic \
   --name "$HF_USERNAME/stembolts"
   --io-yaml io.yaml \
    stembolts_model_3.3_2b
```

## Generate a README

After uploading your adapter, you can auto-generate a README for the HuggingFace model repository using `m alora add-readme`. This command uses Mellea to analyze your training dataset and produce documentation with a description, data examples, and integration code:

```bash
m alora add-readme \
    --name $HF_USERNAME/stembolts \
    --io-yaml io.yaml \
    --basemodel granite-4.0-micro \
    stembolt_failure_dataset.jsonl
```

You can provide a `--hints` file with additional domain context to improve the generated descriptions:

```bash
m alora add-readme \
    --name $HF_USERNAME/stembolts \
    --io-yaml io.yaml \
    --basemodel granite-4.0-micro \
    --hints hints.txt
    stembolt_failure_dataset.jsonl
```

The generator will display the README and ask for confirmation before uploading it to your HuggingFace repo. You can also call the generator programmatically from Python -- see `test_readme_generator.py` for an example.

## Using Intrinsics

You can now create a new adapter class for this model somewhere in your python project:

```python
from mellea.backends.adapters.adapter import CustomGraniteCommonAdapter

class StemboltAdapter(CustomGraniteCommonAdapter):
    def __init__(self, base_model_name:str="granite-4.0-micro"):
        super().__init__(
            model_id="$USERNAME/stembolts", # REPLACE $USERNAME WITH YOUR HUGGINGFACE USERNAME
            intrinsic_name="stembolts",
            base_model_name=base_model_name,
        )
```

Using this adapter requires adding it to a backend:

```python
from mellea.backends.huggingface import LocalHFBackend

backend = LocalHFBackend(
    model_id="ibm-granite/granite-4.0-micro", cache=SimpleLRUCache(5)
)

backend.add_adapter(StemboltAdapter(base_model_name="granite-4.0-micro"))
```

A full example of how to use this adapter as a requirement is found in `101_example.py`

