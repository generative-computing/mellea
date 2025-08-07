<img src="docs/mellea_draft_logo_300.png" height=100>

# Mellea

Mellea is a library for writing generative programs.
Generative programming replaces flaky agents and brittle prompts
with structured, maintainable, robust, and efficient AI workflows.


[//]: # ([![arXiv]&#40;https://img.shields.io/badge/arXiv-2408.09869-b31b1b.svg&#41;]&#40;https://arxiv.org/abs/2408.09869&#41;)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://mellea.ai/)
[![PyPI version](https://img.shields.io/pypi/v/mellea)](https://pypi.org/project/mellea/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mellea)](https://pypi.org/project/mellea/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![GitHub License](https://img.shields.io/github/license/generative-computing/mellea)](https://img.shields.io/github/license/generative-computing/mellea)



## Features

 * A standard library of opinionated prompting patterns.
 * Sampling strategies for inference-time scaling.
 * Clean integration between verifiers and samplers.
    - Batteries-included library of verifiers.
    - Support for efficient checking of specialized requirements using
      activated LoRAs.
    - Train your own verifiers on proprietary classifier data.
 * Compatible with many inference services and model families. Control cost
   and quality by easily lifting and shifting workloads between:
        - inference providers
        - model families
        - model sizes
 * Easily integrate the power of LLMs into legacy code-bases (mify).
 * Sketch applications by writing specifications and letting `mellea` fill in
   the details (generative slots).
 * Get started by decomposing your large unwieldy prompts into structured and maintainable mellea problems.



## Getting Started

You can get started with a local install, or by using Colab notebooks.

### Getting Started with Local Infernece

<img src="docs/GetStarted_py.png" style="max-width:800px">

Install with pip:

```bash
uv pip install mellea
```

> [!NOTE]
> If running on an Intel mac, you may get errors related to torch/torchvision versions. Conda maintains updated versions of these packages. You will need to create a conda environment and run `conda install 'torchvision>=0.22.0'` (this should also install pytorch and torchvision-extra). Then, you should be able to run `uv pip install mellea`. To run the examples, you will need to use `python <filename>` inside the conda environment instead of `uv run --with mellea <filename>`.

> [!NOTE]
> If you are using python >= 3.13, you may encounter an issue where outlines cannot be installed due to rust compiler issues (`error: can't find Rust compiler`). You can either downgrade to python 3.12 or install the [rust compiler](https://www.rust-lang.org/tools/install) to build the wheel for outlines locally.

For running a simple LLM request locally (using Ollama with Granite model), this is the starting code:
```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/example.py
import mellea

m = mellea.start_session()
print(m.chat("What is the etymology of mellea?").content)
```


Then run it:
> [!NOTE]
> Before we get started, you will need to download and install [ollama](https://ollama.com/). Mellea can work with many different types of backends, but everything in this tutorial will "just work" on a Macbook running IBM's Granite 3.3 8B model.
```shell
uv run --with mellea docs/examples/tutorial/example.py
```

### Get Started with Colab

| Notebook | Try in Colab | Goal |
|----------|--------------|------|
| Hello, World | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Quick‑start demo |
| Simple Email | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/simple_email.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Using the `m.instruct` primitive |
| Instruct-Validate-Repair | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/instruct_validate_repair.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Introduces our first generative programming design pattern |
| Model Options | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notsbooks/model_options_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Demonstrates how to pass model options through to backends |
| Sentiment Classifier | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/sentiment_classifier.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Introduces the `@generative` decorator |
| Managing Context | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main//docs/examples/notebooks/context_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Shows how to construct and manage context in a `MelleaSession` |
| Generative OOP | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/table_mobject.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Demonstrates object-oriented generative programming in Mellea |
| Rich Documents | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/document_mobject.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | A generative program that uses Docling to work with rich-text documents |
| Composing Generative Functions | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/compositionality_with_generative_slots.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Demonstrates contract-oriented programming in Mellea |
| `m serve` | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/m_serve_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Serve a generative program as an openai-compatible model endpoint |
| MCP | <a href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/mcp_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Mellea + MCP |


### Installing from source

Fork and clone the repositoy:

```bash
git clone ssh://git@github.com/<my-username>/mellea.git && cd mellea/
```

Setup a virtual environment:

```bash
uv venv .venv && source .venv/bin/activate
```

Use `uv pip` to install from source with the editable flag:

```bash
uv pip install -e .
```

Ensure that you install the precommit hooks:

```bash
pre-commit install
```

## Getting started with validation

Mellea supports validation of generation results through a **instruct-validate-repair** pattern.
Below, the request for *"Write an email.."* is constrained by the requirements of *"be formal"* and *"Use 'Dear interns' as greeting."*.
Using a simple rejection sampling strategy, the request is sent up to three (loop_budget) times to the model and
the output is checked against the constraints using (in this case) LLM-as-a-judge.


```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/instruct_validate_repair/101_email_with_validate.py
from mellea import MelleaSession
from mellea.backends.types import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import model_ids
from mellea.stdlib.sampling import RejectionSamplingStrategy

# create a session with Mistral running on Ollama
m = MelleaSession(
    backend=OllamaModelBackend(
        model_id=model_ids.MISTRALAI_MISTRAL_0_3_7B,
        model_options={ModelOption.MAX_NEW_TOKENS: 300},
    )
)

# run an instruction with requirements
email_v1 = m.instruct(
    "Write an email to invite all interns to the office party.",
    requirements=["be formal", "Use 'Dear interns' as greeting."],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)

# print result
print(f"***** email ****\n{str(email_v1)}\n*******")
```


## Getting Started with Generative Slots

Generative slots allow you to define functions without implementing them.
The `@generative` decorator marks a function as one that should be interpreted by querying an LLM.
The example below demonstrates how an LLM's sentiment classification
capability can be wrapped up as a function using Mellea's generative slots and
a local LLM.


```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/sentiment_classifier.py#L1-L13
from typing import Literal
from mellea import generative, start_session


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
  """Classify the sentiment of the input text as 'positive' or 'negative'."""


if __name__ == "__main__":
  m = start_session()
  sentiment = classify_sentiment(m, text="I love this!")
  print("Output sentiment is:", sentiment)
```



## Tutorial

See the [tutorial](docs/tutorial.md)

## Contributing

Please refer to the [Contributor Guide](docs/tutorial.md#appendix-contributing-to-mellea) for detailed instructions on how to contribute.

### IBM ❤️ Open Source AI

Mellea has been started by IBM Research in Cambridge, MA.



