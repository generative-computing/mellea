---
title: "Installation"
description: "Install Mellea and set up your Python environment."
# diataxis: tutorial
---

# Installation

**Prerequisites:** Python 3.10+, `pip` or `uv` available.

## Install

```bash
pip install mellea
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add mellea
```

## Optional extras

Install extras for specific backends:

```bash
pip install mellea[litellm]    # LiteLLM multi-provider (Anthropic, Bedrock, etc.)
pip install mellea[hf]         # HuggingFace transformers for local inference
pip install mellea[watsonx]    # IBM WatsonX
pip install mellea[tools]      # Tool and agent dependencies
pip install mellea[telemetry]  # OpenTelemetry tracing and metrics
```

You can combine extras:

```bash
pip install mellea[litellm,tools,telemetry]
```

## Default backend: Ollama

The default session connects to [Ollama](https://ollama.ai) running locally.
Install Ollama and pull the default model before running any examples:

```bash
ollama pull granite4:micro
```
