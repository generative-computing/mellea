<img src="https://github.com/generative-computing/mellea/raw/main/docs/mellea_draft_logo_300.png" alt="Mellea logo" height=100>

# Mellea — build predictable AI without guesswork

Inside every AI-powered pipeline, the unreliable part is the same: the LLM call itself.
Silent failures, untestable outputs, no guarantees.
Mellea is a Python library for writing *generative programs* — replacing brittle prompts and flaky agents
with structured, testable AI workflows built around type-annotated outputs, verifiable requirements, and automatic retries.

[//]: # ([![arXiv]&#40;https://img.shields.io/badge/arXiv-2408.09869-b31b1b.svg&#41;]&#40;https://arxiv.org/abs/2408.09869&#41;)
[![Website](https://img.shields.io/badge/website-mellea.ai-blue)](https://mellea.ai/)
[![Docs](https://img.shields.io/badge/docs-docs.mellea.ai-brightgreen)](https://docs.mellea.ai/)
[![PyPI version](https://img.shields.io/pypi/v/mellea)](https://pypi.org/project/mellea/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mellea)](https://pypi.org/project/mellea/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![GitHub License](https://img.shields.io/github/license/generative-computing/mellea)](https://github.com/generative-computing/mellea/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-3.0-4baaaa.svg)](CODE_OF_CONDUCT.md)

## Install

```bash
uv pip install mellea
```

See [installation docs](https://docs.mellea.ai/getting-started/installation) for additional options, such as installing all extras via `uv pip install 'mellea[all]'`.
For source installation directly from this repo, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Example

The `@generative` decorator turns a typed Python function into a structured LLM call.
Docstrings become prompts, type hints become schemas — no templates, no parsers:

```python
from pydantic import BaseModel
from mellea import generative, start_session

class UserProfile(BaseModel):
    name: str
    age: int

@generative
def extract_user(text: str) -> UserProfile:
    """Extract the user's name and age from the text."""

m = start_session()
user = extract_user(m, text="User log 42: Alice is 31 years old.")
print(user.name)  # Alice
print(user.age)   # 31 — always an int, guaranteed by the schema
```

## What Mellea Does

- **Structured output** — `@generative` turns typed functions into LLM calls; Pydantic schemas are enforced at generation time
- **Requirements & repair** — attach natural-language requirements to any call; Mellea validates and retries automatically
- **Sampling strategies** — run a generation multiple times and pick the best result; swap between rejection sampling, majority voting, and more with one parameter change
- **Multiple backends** — Ollama, OpenAI, HuggingFace, WatsonX, LiteLLM, Bedrock
- **Legacy integration** — easily drop Mellea into existing codebases with `mify`
- **MCP compatible** — expose any generative program as an MCP tool

## Learn More

| Resource | Description |
| --- | --- |
| [docs.mellea.ai](https://docs.mellea.ai) | Full docs — vision, tutorials, API reference, how-to guides |
| [Colab notebooks](docs/examples/notebooks/) | Interactive examples you can run immediately |
| [Code examples](docs/examples/) | Runnable examples: RAG, agents, Instruct-Validate-Repair (IVR), MObjects, and more |

## Contributing

We welcome contributions of all kinds — bug fixes, new backends, standard library components, examples, and docs.

- **[Contributing Guide](CONTRIBUTING.md)** — development setup, workflow, and coding standards
- **[Building Extensions](https://docs.mellea.ai/community/building-extensions)** — create reusable components in your own repo
- **[mellea-contribs](https://github.com/generative-computing/mellea-contribs)** — community library for shared components

Questions? See [GitHub Discussions](https://github.com/generative-computing/mellea/discussions).

### IBM ❤️ Open Source AI

Mellea was started by IBM Research in Cambridge, MA.

---

Licensed under the [Apache-2.0 License](LICENSE). Copyright © 2026 Mellea.
