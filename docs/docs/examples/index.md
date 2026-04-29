---
canonical: "https://docs.mellea.ai/examples/index"
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

## All example categories

The repository contains many more runnable examples than the four documented
above. Every category has its own `README.md` and one or more `.py` files ready
to run.

### Core concepts

| Category | What it shows |
| -------- | ------------- |
| `instruct_validate_repair/` | The IVR loop end-to-end: basic generation, adding requirements, automatic repair on failure, custom validators |
| `generative_stubs/` | `@generative` functions with typed returns, pipeline composition, `ChatContext` persona injection, pre/postcondition checks |
| `context/` | Context inspection, sampling with context trees, parallel context branches |
| `sessions/` | Custom session types and backend selection |
| `async/` | How to utilize basic async capabilities |

### Data and documents

| Category | What it shows |
| -------- | ------------- |
| `information_extraction/` | Named entity recognition and type-safe structured extraction with Pydantic |
| `mobject/` | Table queries and transformations using `MObject` structured data types |
| `mify/` | `@mify` on existing classes — custom string representations, field filtering, `funcs_include` |
| `rag/` | FAISS vector search, `@generative bool` relevance filter, `grounding_context` for grounded generation |

### Agents and tools

| Category | What it shows |
| -------- | ------------- |
| `agents/` | ReACT reasoning-and-acting loop, multi-turn tool workflows |
| `tools/` | `@tool` definition, code interpreter integration, tool argument validation, safe `eval` patterns |
| `mini_researcher/` | Complete research assistant: multi-model architecture, document retrieval, safety checks, custom validation pipeline |

### Extensibility

| Category | What it shows |
| -------- | ------------- |
| `plugins/` | Plugin system end-to-end: function hooks, class-based plugins, payload modification, scoped and session-scoped plugins, `PluginSet` composition, execution modes, tool hooks, and testing patterns |

### Safety and validation

| Category | What it shows |
| -------- | ------------- |
| `safety/` | `GuardianCheck` for harm, jailbreak, profanity, social bias, violence, and groundedness; shared backend pattern |

### Integration and deployment

| Category | What it shows |
| -------- | ------------- |
| `m_serve/` | Deploying Mellea programs as REST APIs with production deployment patterns |
| `m_decompose/` | Decomposing complex prompts into sub-tasks via the CLI and Python API |
| `library_interop/` | LangChain message conversion, OpenAI format compatibility, cross-library workflows |
| `mcp/` | MCP tool creation, Claude Desktop integration, Langflow integration |
| `bedrock/` | Amazon Bedrock backend configuration and usage |

### Performance and advanced sampling

| Category | What it shows |
| -------- | ------------- |
| `aLora/` | Training aLoRA adapters for fast constraint checking; performance optimisation |
| `intrinsics/` | Answer relevance, hallucination detection, citation validation, context relevance — specialised adapter-backed checks |
| `sofai/` | Two-tier sampling: fast-model iteration with escalation to a slow model; cost optimisation |

### Multimodal

| Category | What it shows |
| -------- | ------------- |
| `image_text_models/` | Vision-language models, `ImageBlock`, multimodal prompting, backend support matrix |

### Observability

| Category | What it shows |
| -------- | ------------- |
| `telemetry/` | OpenTelemetry application and backend traces; span export configuration |

### Experimental

| Category | What it shows |
| -------- | ------------- |
| `melp/` | ⚠️ Experimental lazy evaluation — thunks, deferred execution, advanced control flow |

### Getting started and tutorials

| Category | What it shows |
| -------- | ------------- |
| `hello_world.py` | Minimal single-file starting point |
| `tutorial/` | Python script versions of the tutorials: email generation, IVR, generative stubs, contexts, MObjects, model options, and more |
| `notebooks/` | Jupyter notebook versions of the same tutorials for interactive, cell-by-cell exploration |

---

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
