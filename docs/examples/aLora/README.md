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

If model training succeeds, you will need to upload your model as an adapter function:

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

```
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

After uploading your adapter, you can auto-generate a README for the Hugging Face model repository using `m alora add-readme`. This command uses Mellea to analyze your training dataset and produce documentation with a description, data examples, and integration code:

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
    --hints hints.txt
    stembolt_failure_dataset.jsonl
```

The generator will display the README and ask for confirmation before uploading it to your Hugging Face repo. You can also call the generator programmatically from Python — see `example_readme_generator.py` for an example.

## Using Adapter Functions

Compose the adapter directly from an `Identity`, an output contract, and a
weights binding — no dedicated adapter class needed:

```python
from mellea.backends.adapters import Adapter, Identity, LocalFileBinding, get_io_contract
from mellea.backends.adapters.catalog import AdapterType

def stembolt_adapter(base_model_name: str = "granite-4.1-3b") -> Adapter:
    return Adapter(
        identity=Identity(name="stembolts", adapter_type="alora"),
        # get_io_contract falls back to a permissive dict contract for names
        # outside the built-in catalog.
        io_contract=get_io_contract("stembolts"),
        weights=LocalFileBinding(
            name="stembolts",
            adapter_type=AdapterType.ALORA,
            repo_id="$USERNAME/stembolts",  # REPLACE $USERNAME WITH YOUR HUGGING FACE USERNAME
            # A custom, non-catalog adapter has no catalog entry to resolve
            # revision=None from — pin it explicitly.
            revision="main",
        ),
    )
```

Registering it with a backend downloads and loads the weights:

```python
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.huggingface import LocalHFBackend

backend = LocalHFBackend(
    model_id="ibm-granite/granite-4.1-3b", cache=SimpleLRUCache(5)
)

backend.add_adapter(stembolt_adapter(base_model_name="granite-4.1-3b"))
```

## Example Files

### 101_example.py
Comparing aLoRA-backed requirement validation against LLM-as-a-judge, using the
catalog-native `requirement-check` adapter (not a custom one — see
`stembolts_intrinsic.py` for loading your own).

Demonstrates:
- Registering a catalog adapter as an aLoRA
- `ALoraRequirement` vs `LLMaJRequirement` routing
- Timing both validation paths

### 102_example.py
Interactive loop that exercises the custom `stembolts` adapter — skip-marked
because it blocks on stdin in an infinite loop.

The example uses Granite 3.3 2B because that is the available `stembolts`
adapter variant; Granite 4.0 and 4.1 variants are not currently available.

Demonstrates:
- Loading a fully custom, non-catalog adapter via `stembolts_intrinsic.py`
- Running the adapter-backed intrinsic repeatedly with a fresh `ChatContext`

### make_training_data.py
Utility for preparing training datasets for adapter training.

Demonstrates:
- Dataset format preparation
- Data validation
- Preprocessing for training

### stembolts_intrinsic.py
Helper module for loading and calling the fully custom `stembolts` adapter
(no standalone entry point — `102_example.py` consumes it).

Demonstrates:
- Custom adapter class definition (`StemboltAdapter`)
- Registering the adapter on a backend, guarded by qualified name
- Calling the intrinsic directly via `mfuncs.act`

### example_readme_generator.py
Programmatic README generation for adapters.

Demonstrates:
- Auto-generating adapter documentation
- Using the `m alora add-readme` command programmatically
- Documenting custom adapters
