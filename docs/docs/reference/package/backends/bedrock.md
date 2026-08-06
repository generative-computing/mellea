---
id: bedrock
title: "mellea.backends.bedrock"
sidebar_label: "bedrock"
sidebar_position: 3
description: "Helpers for creating bedrock backends from openai/litellm."
# diataxis: reference
---

Source: [`mellea/backends/bedrock.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/bedrock.py) at commit `a535fc6345a0`.

Helpers for creating bedrock backends from openai/litellm.

## `list_mantle_models()`

*function* — [line 93](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/bedrock.py#L93)

`list_mantle_models(region: str | None = None) -> list`

Return all models available at a bedrock-mantle endpoint.

## `stringify_mantle_model_ids()`

*function* — [line 111](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/bedrock.py#L111)

`stringify_mantle_model_ids(region: str | None = None) -> str`

Return a human-readable list of all models available at the mantle endpoint for an AWS region.

## `create_bedrock_litellm_backend()`

*function* — [line 125](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/bedrock.py#L125)

`create_bedrock_litellm_backend(model_id: ModelIdentifier | str, region: str | None = None, num_retries: int = 3) -> LiteLLMBackend`

Returns a LiteLLM backend that points to Bedrock for model `model_id`.

## `create_bedrock_openai_backend()`

*function* — [line 176](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/bedrock.py#L176)

`create_bedrock_openai_backend(model_id: ModelIdentifier | str, region: str | None = None) -> OpenAIBackend`

Return an OpenAI backend that points to Bedrock mantle for the given model.

## `create_bedrock_mantle_backend()`

*function* — [line 238](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/bedrock.py#L238)

`create_bedrock_mantle_backend(model_id: ModelIdentifier | str, region: str | None = None) -> OpenAIBackend`

Deprecated alias for `create_bedrock_openai_backend`.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
