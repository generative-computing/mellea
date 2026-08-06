---
id: start_backend
title: "mellea.stdlib.start_backend"
sidebar_label: "start_backend"
sidebar_position: 9
description: "Typed `start_backend` with overloaded return types."
# diataxis: reference
---

Source: [`mellea/stdlib/start_backend.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/start_backend.py) at commit `a535fc6345a0`.

Typed `start_backend` with overloaded return types.

## `backend_name_to_class()`

*function* — [line 26](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/start_backend.py#L26)

`backend_name_to_class(name: str) -> Any`

Resolves backend names to Backend classes.

## `start_backend()`

*function* — [line 316](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/start_backend.py#L316)

`start_backend(backend_name: Literal['ollama', 'hf', 'openai', 'watsonx', 'litellm'] = 'ollama', model_id: str | ModelIdentifier = IBM_GRANITE_4_MICRO_3B, ctx: Context | None = None, *, context_type: Literal['simple', 'chat'] | None = None, model_options: dict | None = None, **backend_kwargs: Any) -> tuple[Context, Backend]`

Create a context and backend pair without a full session.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
