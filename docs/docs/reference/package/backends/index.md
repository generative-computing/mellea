---
id: index
title: "mellea.backends"
sidebar_label: "Overview"
sidebar_position: 0
description: "Backend implementations for the mellea inference layer."
# diataxis: reference
---

Source: [`mellea/backends/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/__init__.py) at commit `a535fc6345a0`.

Backend implementations for the mellea inference layer.

Declared exports (`__all__`): `Backend`, `BaseModelSubclass`, `FormatterBackend`, `MelleaTool`, `ModelIdentifier`, `ModelOption`, `SimpleLRUCache`, `tool`

## Modules

- [`mellea.backends.adapters`](adapters.md) — Classes and Functions for Backend Adapters.
- [`mellea.backends.backend`](backend.md) — `FormatterBackend`: base class for prompt-engineering backends.
- [`mellea.backends.bedrock`](bedrock.md) — Helpers for creating bedrock backends from openai/litellm.
- [`mellea.backends.cache`](cache.md) — Cache abstractions and implementations for model state.
- [`mellea.backends.context_lengths`](context_lengths.md) — Model context-length lookup table.
- [`mellea.backends.dummy`](dummy.md) — This module holds shim backends used for smoke tests.
- [`mellea.backends.huggingface`](huggingface.md) — A backend that uses the Hugging Face Transformers library.
- [`mellea.backends.kv_block_helpers`](kv_block_helpers.md) — Low-level utilities for concatenating transformer KV caches (KV smashing).
- [`mellea.backends.litellm`](litellm.md) — A generic LiteLLM compatible backend that wraps around the openai python sdk.
- [`mellea.backends.model_ids`](model_ids.md) — `ModelIdentifier` dataclass and a catalog of pre-defined model IDs.
- [`mellea.backends.model_options`](model_options.md) — Common ModelOptions for Backend Generation.
- [`mellea.backends.ollama`](ollama.md) — A model backend wrapping the Ollama Python SDK.
- [`mellea.backends.openai`](openai.md) — A generic OpenAI compatible backend that wraps around the openai python sdk.
- [`mellea.backends.tools`](tools.md) — LLM tool definitions, parsing, and validation for mellea backends.
- [`mellea.backends.utils`](utils.md) — Shared utility functions used across formatter-based backend implementations.
- [`mellea.backends.watsonx`](watsonx.md) — A generic WatsonX.ai compatible backend that wraps around the watson_machine_learning library.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
