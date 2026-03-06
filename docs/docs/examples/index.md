---
title: "Examples"
description: "Complete working programs demonstrating Mellea patterns in production-like scenarios."
# diataxis: reference
---

Each example in this section is a complete, runnable Python program. The pages
walk through the code section by section so you can see how the pieces fit
together. Copy any example as a starting point for your own project.

## Examples in this section

| Example | What it shows |
| ------- | ------------- |
| [Data extraction pipeline](./data-extraction-pipeline) | Use `@generative` with a typed return to pull structured data from unstructured text |
| [Legacy code integration](./legacy-code-integration) | Apply `@mify` to existing Python classes so the model can act on them |
| [Resilient RAG with fallback](./resilient-rag-fallback) | Build a FAISS retrieval pipeline with an LLM relevance filter before generation |
| [Traced generation loop](./traced-generation-loop) | Enable OpenTelemetry application and backend traces with two environment variables |

## Running the examples

All examples are in the `docs/examples/` directory of the repository. Unless
otherwise noted, run them with:

```bash
python docs/examples/<folder>/<file>.py
```

Some examples declare inline script dependencies using the
[PEP 723](https://peps.python.org/pep-0723/) `/// script` block and can be
run with `uv run` instead:

```bash
uv run docs/examples/<folder>/<file>.py
```

**Default backend:** `start_session()` with no arguments connects to a local
[Ollama](https://ollama.ai) instance running **IBM Granite 4 Micro**
(`granite4:micro`). Make sure Ollama is running before you execute any example.
