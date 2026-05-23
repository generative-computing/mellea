---
canonical: "https://docs.mellea.ai/getting-started/installation"
title: "Installation"
description: "Install Mellea and set up your Python environment."
# diataxis: tutorial
---

**Prerequisites:** Python 3.11+, [pip](https://pip.pypa.io/) or [uv](https://docs.astral.sh/uv/) available.

## Install

For an existing Python environment, install Mellea with pip:

```bash
pip install mellea
```

For an existing uv project, add Mellea as a project dependency:

```bash
uv add mellea
```

If you are starting from scratch with uv, initialize the project first so uv can
create `pyproject.toml`, select a supported Python version, and manage the
virtual environment:

```bash
uv init my-mellea-app --python 3.11
cd my-mellea-app
uv add mellea
```

If you only want to try Mellea inside the current virtual environment without
creating a uv project, use `uv pip install` instead:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install mellea
```

## Optional extras

Install extras for specific backends and features:

```bash
pip install "mellea[litellm]"    # LiteLLM multi-provider (Anthropic, Bedrock, etc.)
pip install "mellea[hf]"         # HuggingFace transformers for local inference
pip install "mellea[watsonx]"    # IBM WatsonX
pip install "mellea[tools]"      # Tool and agent dependencies (LangChain, smolagents)
pip install "mellea[cli]"        # m serve, m alora, m decompose CLI commands
pip install "mellea[telemetry]"  # OpenTelemetry tracing and metrics
```

```bash
uv add "mellea[litellm]"        # LiteLLM multi-provider (Anthropic, Bedrock, etc.)
uv add "mellea[hf]"             # HuggingFace transformers for local inference
uv add "mellea[watsonx]"        # IBM WatsonX
uv add "mellea[tools]"          # Tool and agent dependencies (LangChain, smolagents)
uv add "mellea[cli]"            # m serve, m alora, m decompose CLI commands
uv add "mellea[telemetry]"      # OpenTelemetry tracing and metrics
```

You can combine extras:

```bash
pip install "mellea[litellm,tools,telemetry]"
```

```bash
uv add "mellea[litellm,tools,telemetry]"
```

> **All extras:** `mellea[all]` installs everything. For the full list of available
> extras see [`pyproject.toml`](https://github.com/generative-computing/mellea/blob/main/pyproject.toml).

## Default backend: Ollama

The default session connects to [Ollama](https://ollama.ai) running locally.
Install Ollama and pull the default model before running any examples:

```bash
ollama pull granite4.1:3b
```
