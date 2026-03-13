---
title: "Installation"
description: "Install Mellea and set up your Python environment."
# diataxis: tutorial
---

**Prerequisites:** Python 3.11+, [pip](https://pip.pypa.io/) or [uv](https://docs.astral.sh/uv/) available.

## Install

```bash
pip install mellea
```

```bash
uv add mellea
```

## Optional extras

Install extras for specific backends and features:

```bash
pip install "mellea[litellm]"    # LiteLLM multi-provider (Anthropic, Bedrock, etc.)
pip install "mellea[hf]"         # HuggingFace transformers for local inference
pip install "mellea[watsonx]"    # IBM WatsonX
pip install "mellea[tools]"      # Tool and agent dependencies (LangChain, smolagents)
pip install "mellea[telemetry]"  # OpenTelemetry tracing and metrics
```

```bash
uv add "mellea[litellm]"        # LiteLLM multi-provider (Anthropic, Bedrock, etc.)
uv add "mellea[hf]"             # HuggingFace transformers for local inference
uv add "mellea[watsonx]"        # IBM WatsonX
uv add "mellea[tools]"          # Tool and agent dependencies (LangChain, smolagents)
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
ollama pull granite4:micro
```
